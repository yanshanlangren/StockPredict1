#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态预测模型 - 融合新闻、文本、技术指标、相关性特征。

Phase 4 增强：
1. 新增文本向量分支（统计特征 + Hashing 向量）
2. 支持更重模型结构（heavy）
3. 与结构化基线自动做增量对比，未优于基线时仅保存候选模型
"""

import os
import json
import logging
import hashlib
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入 TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam

    TENSORFLOW_AVAILABLE = True
    logger.info("✓ TensorFlow已加载，多模态模型功能可用")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow未安装，训练功能不可用，预测将降级为规则模式")

# 获取项目根目录
def _get_project_root():
    """获取项目根目录"""
    if os.environ.get("COZE_WORKSPACE_PATH"):
        return os.environ.get("COZE_WORKSPACE_PATH")
    try:
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file)
        return os.path.dirname(src_dir)
    except Exception:
        return os.getcwd()


PROJECT_ROOT = _get_project_root()
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_NAME = "multimodal_stock_predictor"


class MultiModalPredictor:
    """多模态股票预测模型（单例）"""

    _instance = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_dir: str = None):
        """初始化多模态预测器"""
        if model_dir is None:
            model_dir = MODEL_DIR

        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.model = None
        self.model_name = MODEL_NAME
        self.metadata: Dict[str, Any] = {}

        # 特征维度
        self.news_feature_dim = 64
        self.sector_feature_dim = 32
        self.tech_feature_dim = 12
        self.relevance_dim = 16
        self.text_feature_dim = 128
        self.text_stat_dim = 16
        self.text_hash_dim = self.text_feature_dim - self.text_stat_dim

        # 领域列表
        self.sectors = [
            "银行",
            "证券",
            "保险",
            "地产",
            "汽车",
            "科技",
            "医药",
            "消费",
            "能源",
            "军工",
            "基建",
            "传媒",
            "教育",
            "农业",
            "交通",
        ]

        self.positive_keywords = ["增长", "利好", "上涨", "增持", "突破", "回购", "盈利", "超预期", "高景气", "订单"]
        self.negative_keywords = ["下滑", "利空", "下跌", "减持", "亏损", "风险", "处罚", "违约", "暴雷", "裁员"]

        # 尝试加载模型
        self._load_model()

    def is_available(self) -> bool:
        """检查模型是否可用。规则版预测始终可用。"""
        return True

    def _keras_model_path(self) -> str:
        return os.path.join(self.model_dir, f"{self.model_name}.keras")

    def _keras_candidate_path(self) -> str:
        return os.path.join(self.model_dir, f"{self.model_name}_candidate.keras")

    def _metadata_path(self) -> str:
        return os.path.join(self.model_dir, f"{self.model_name}_metadata.json")

    def _load_model(self):
        """加载已训练模型与元数据"""
        metadata_path = self._metadata_path()
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as file_obj:
                    self.metadata = json.load(file_obj)
            except Exception as e:
                logger.warning(f"读取模型元数据失败: {e}")

        if TENSORFLOW_AVAILABLE:
            model_path = self._keras_model_path()
            if os.path.exists(model_path):
                try:
                    self.model = keras.models.load_model(model_path)
                    logger.info(f"✓ 多模态 TensorFlow 模型加载成功: {model_path}")
                except Exception as e:
                    logger.warning(f"加载 TensorFlow 模型失败: {e}")
                    self.model = None

        if self.model is None:
            logger.info("未找到已训练多模态模型，当前使用规则版预测")

    def _build_model(self, model_tier: str = "heavy"):
        """构建多模态神经网络模型（固定 heavy 结构）"""
        if not TENSORFLOW_AVAILABLE:
            return None

        # 模型复杂度固定为 heavy；model_tier 参数仅为兼容旧调用链
        _ = model_tier
        news_units, sector_units, tech_units, relevance_units, text_units = 64, 32, 64, 24, 128
        fusion_layers = [256, 128, 64]
        fusion_dropouts = [0.35, 0.25, 0.15]
        learning_rate = 8e-4

        # 新闻特征输入
        news_input = layers.Input(shape=(self.news_feature_dim,), name="news_features")
        news_branch = layers.Dense(news_units, activation="relu")(news_input)
        news_branch = layers.Dropout(0.2)(news_branch)

        # 领域影响输入
        sector_input = layers.Input(shape=(self.sector_feature_dim,), name="sector_features")
        sector_branch = layers.Dense(sector_units, activation="relu")(sector_input)

        # 技术指标输入
        tech_input = layers.Input(shape=(self.tech_feature_dim,), name="tech_features")
        tech_branch = layers.Dense(tech_units, activation="relu")(tech_input)
        tech_branch = layers.Dropout(0.2)(tech_branch)

        # 相关性输入
        relevance_input = layers.Input(shape=(self.relevance_dim,), name="relevance_features")
        relevance_branch = layers.Dense(relevance_units, activation="relu")(relevance_input)

        # 文本向量输入（Phase 4）
        text_input = layers.Input(shape=(self.text_feature_dim,), name="text_features")
        text_branch = layers.Dense(text_units, activation="relu")(text_input)
        text_branch = layers.Dropout(0.25)(text_branch)

        # 融合
        concat = layers.Concatenate()(
            [news_branch, sector_branch, tech_branch, relevance_branch, text_branch]
        )

        x = concat
        for layer_units, layer_dropout in zip(fusion_layers, fusion_dropouts):
            x = layers.Dense(layer_units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(layer_dropout)(x)

        classification_output = layers.Dense(1, activation="sigmoid", name="prediction")(x)

        # 真实收益回归头（目标为 future_ret_5d，十进制收益率）
        return_hidden = layers.Dense(32, activation="relu")(x)
        return_output = layers.Dense(1, activation="linear", name="return_pred")(return_hidden)

        model = models.Model(
            inputs=[news_input, sector_input, tech_input, relevance_input, text_input],
            outputs=[classification_output, return_output],
        )

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                "prediction": "binary_crossentropy",
                "return_pred": keras.losses.Huber(delta=0.02),
            },
            loss_weights={
                "prediction": 1.0,
                "return_pred": 0.35,
            },
            metrics={
                "prediction": [
                    keras.metrics.BinaryAccuracy(name="accuracy"),
                    keras.metrics.AUC(name="auc"),
                ],
                "return_pred": [
                    keras.metrics.MeanAbsoluteError(name="mae"),
                    keras.metrics.MeanSquaredError(name="mse"),
                ],
            },
        )
        return model

    def _normalize_2d(self, array: np.ndarray, expected_dim: int) -> np.ndarray:
        """将输入标准化为二维矩阵，并按目标维度截断/补零。"""
        arr = np.asarray(array, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)

        current_dim = arr.shape[1]
        if current_dim == expected_dim:
            return arr

        if current_dim > expected_dim:
            return arr[:, :expected_dim]

        pad = np.zeros((arr.shape[0], expected_dim - current_dim), dtype=float)
        return np.hstack([arr, pad])

    def _prepare_training_arrays(self, training_data: Dict) -> Dict[str, np.ndarray]:
        required_keys = [
            "news_features",
            "sector_features",
            "tech_features",
            "relevance_features",
            "labels",
        ]
        missing_keys = [key for key in required_keys if key not in training_data]
        if missing_keys:
            raise ValueError(f"训练数据缺少字段: {', '.join(missing_keys)}")

        labels = np.asarray(training_data["labels"]).astype(int).reshape(-1)
        n_samples = len(labels)
        if n_samples == 0:
            raise ValueError("训练数据为空")

        news = self._normalize_2d(training_data["news_features"], self.news_feature_dim)
        sector = self._normalize_2d(training_data["sector_features"], self.sector_feature_dim)
        tech = self._normalize_2d(training_data["tech_features"], self.tech_feature_dim)
        relevance = self._normalize_2d(training_data["relevance_features"], self.relevance_dim)

        text_raw = training_data.get("text_features")
        if text_raw is None:
            text = np.zeros((n_samples, self.text_feature_dim), dtype=float)
        else:
            text = self._normalize_2d(text_raw, self.text_feature_dim)

        return_targets_raw = training_data.get("return_targets")
        if return_targets_raw is None:
            logger.warning("训练数据缺少 return_targets，回归头将使用标签近似值回退")
            return_targets = np.where(labels > 0, 0.01, -0.01).astype(float)
        else:
            return_targets = np.asarray(return_targets_raw, dtype=float).reshape(-1)
            if len(return_targets) != n_samples:
                raise ValueError(
                    f"return_targets 样本数与标签不一致: {len(return_targets)} != {n_samples}"
                )

        # 过滤异常值并裁剪至合理区间（future_ret_5d）
        return_targets = np.nan_to_num(return_targets, nan=0.0, posinf=0.0, neginf=0.0)
        return_targets = np.clip(return_targets, -0.20, 0.20)

        matrices = {
            "news": news,
            "sector": sector,
            "tech": tech,
            "relevance": relevance,
            "text": text,
            "labels": labels,
            "return_targets": return_targets.reshape(-1, 1),
        }

        for name, matrix in matrices.items():
            if name == "labels":
                continue
            if matrix.shape[0] != n_samples:
                raise ValueError(f"{name} 样本数与标签不一致: {matrix.shape[0]} != {n_samples}")

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError("训练标签仅包含单一类别，无法训练分类模型")

        return matrices

    def _load_baseline_metrics(self) -> Dict[str, Any]:
        """读取最近结构化基线报告中的关键指标。"""
        report_path = os.path.join(RESULT_DIR, "structured_baseline_report_latest.json")
        if not os.path.exists(report_path):
            return {
                "available": False,
                "source": report_path,
                "message": "未找到 structured_baseline_report_latest.json",
            }

        try:
            with open(report_path, "r", encoding="utf-8") as file_obj:
                report = json.load(file_obj)

            test_metrics = report.get("metrics", {}).get("test", {})
            if not test_metrics:
                holdout = report.get("holdout_results", {}).get("all_features", {})
                test_metrics = holdout.get("test_metrics", {}) if isinstance(holdout, dict) else {}

            return {
                "available": bool(test_metrics),
                "source": report_path,
                "created_at": report.get("created_at"),
                "model_type": report.get("model_type") or report.get("config", {}).get("model_type"),
                "accuracy": self._safe_float(test_metrics.get("accuracy"), None),
                "auc": self._safe_float(test_metrics.get("auc"), None),
                "f1": self._safe_float(test_metrics.get("f1"), None),
                "topk_hit_rate": self._safe_float(test_metrics.get("topk_hit_rate"), None),
            }
        except Exception as e:
            logger.warning(f"读取基线报告失败: {e}")
            return {
                "available": False,
                "source": report_path,
                "message": str(e),
            }

    def _compare_with_baseline(self, final_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """将多模态模型指标与最近基线报告对比。"""
        baseline = self._load_baseline_metrics()
        comparison = {
            "baseline_available": bool(baseline.get("available")),
            "baseline": baseline,
            "target_metric": None,
            "target_value": None,
            "baseline_metric": None,
            "baseline_value": None,
            "delta": None,
            "outperform_baseline": None,
            "message": "",
        }

        if not comparison["baseline_available"]:
            comparison["message"] = "未找到可用基线报告，默认允许替换模型"
            return comparison

        # 优先使用时间顺序 holdout(test) 指标，再回退到验证集指标
        metric_candidates = [
            ("test_prediction_auc", "auc"),
            ("test_auc", "auc"),
            ("test_prediction_accuracy", "accuracy"),
            ("test_accuracy", "accuracy"),
            ("val_prediction_auc", "auc"),
            ("val_auc", "auc"),
            ("val_prediction_accuracy", "accuracy"),
            ("val_accuracy", "accuracy"),
        ]

        for target_metric, baseline_metric in metric_candidates:
            target_value = self._safe_float(final_metrics.get(target_metric), None)
            baseline_value = self._safe_float(baseline.get(baseline_metric), None)
            if target_value is None or baseline_value is None:
                continue

            delta = target_value - baseline_value
            comparison.update(
                {
                    "target_metric": target_metric,
                    "target_value": target_value,
                    "baseline_metric": baseline_metric,
                    "baseline_value": baseline_value,
                    "delta": delta,
                    "outperform_baseline": bool(delta > 0.0),
                    "message": (
                        f"{target_metric}={target_value:.4f}, "
                        f"baseline.{baseline_metric}={baseline_value:.4f}, "
                        f"delta={delta:+.4f}"
                    ),
                }
            )
            return comparison

        comparison["message"] = "基线或当前模型缺少可比指标，默认允许替换模型"
        return comparison

    @staticmethod
    def _should_replace_model(comparison: Dict[str, Any]) -> bool:
        """依据对比结果判断是否替换线上模型。"""
        if comparison.get("baseline_available") and comparison.get("outperform_baseline") is False:
            return False
        return True

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        metadata_path = self._metadata_path()
        with open(metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2, ensure_ascii=False)
        self.metadata = metadata

    def _train_tensorflow(
        self,
        arrays: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int,
        model_tier: str,
    ) -> Dict[str, Any]:
        """使用 TensorFlow 训练多模态模型。"""
        model = self._build_model(model_tier=model_tier)
        X_all = [arrays["news"], arrays["sector"], arrays["tech"], arrays["relevance"], arrays["text"]]
        y_all_cls = arrays["labels"].astype(float).reshape(-1, 1)
        y_all_ret = arrays["return_targets"].astype(float).reshape(-1, 1)
        total_samples = int(len(y_all_cls))

        if total_samples < 32:
            raise ValueError("样本量过少（<32），无法稳定进行时间顺序 train/val/test 切分")

        # 时间顺序切分，避免随机切分造成时序泄漏
        val_size = max(1, int(total_samples * 0.15))
        test_size = max(1, int(total_samples * 0.15))
        train_size = total_samples - val_size - test_size

        if train_size < 16:
            val_size = max(1, int(total_samples * 0.1))
            test_size = max(1, int(total_samples * 0.1))
            train_size = total_samples - val_size - test_size

        if train_size < 16:
            raise ValueError("样本分割后训练集过小，无法继续训练")

        train_end = train_size
        val_end = train_size + val_size

        X_train = [matrix[:train_end] for matrix in X_all]
        X_val = [matrix[train_end:val_end] for matrix in X_all]
        X_test = [matrix[val_end:] for matrix in X_all]

        y_train_cls = y_all_cls[:train_end]
        y_train_ret = y_all_ret[:train_end]
        y_val_cls = y_all_cls[train_end:val_end]
        y_val_ret = y_all_ret[train_end:val_end]
        y_test_cls = y_all_cls[val_end:]
        y_test_ret = y_all_ret[val_end:]

        callback_list = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            ),
        ]

        history = model.fit(
            X_train,
            {
                "prediction": y_train_cls,
                "return_pred": y_train_ret,
            },
            epochs=max(5, int(epochs)),
            batch_size=max(8, int(batch_size)),
            validation_data=(
                X_val,
                {
                    "prediction": y_val_cls,
                    "return_pred": y_val_ret,
                },
            ),
            callbacks=callback_list,
            verbose=1,
            shuffle=False,
        )

        def history_last(candidates: List[str], default: float) -> float:
            for key in candidates:
                values = history.history.get(key)
                if values:
                    return self._safe_float(values[-1], default)
            return default

        test_metrics_raw = model.evaluate(
            X_test,
            {
                "prediction": y_test_cls,
                "return_pred": y_test_ret,
            },
            verbose=0,
            return_dict=True,
        )

        def eval_last(candidates: List[str], default: float) -> float:
            for key in candidates:
                if key in test_metrics_raw:
                    return self._safe_float(test_metrics_raw.get(key), default)
            return default

        final_metrics = {
            "loss": history_last(["loss"], 0.0),
            "prediction_loss": history_last(["prediction_loss"], 0.0),
            "return_loss": history_last(["return_pred_loss"], 0.0),
            "accuracy": history_last(["prediction_accuracy", "accuracy"], 0.0),
            "auc": history_last(["prediction_auc", "auc"], 0.5),
            "val_loss": history_last(["val_loss"], 0.0),
            "val_prediction_loss": history_last(["val_prediction_loss"], 0.0),
            "val_return_loss": history_last(["val_return_pred_loss"], 0.0),
            "val_accuracy": history_last(["val_prediction_accuracy", "val_accuracy"], 0.0),
            "val_auc": history_last(["val_prediction_auc", "val_auc"], 0.5),
            "return_mae": history_last(["return_pred_mae", "mae"], 0.0),
            "return_mse": history_last(["return_pred_mse", "mse"], 0.0),
            "val_return_mae": history_last(["val_return_pred_mae", "val_mae"], 0.0),
            "val_return_mse": history_last(["val_return_pred_mse", "val_mse"], 0.0),
            "test_loss": eval_last(["loss"], 0.0),
            "test_prediction_loss": eval_last(["prediction_loss"], 0.0),
            "test_return_loss": eval_last(["return_pred_loss"], 0.0),
            "test_accuracy": eval_last(["prediction_accuracy", "accuracy"], 0.0),
            "test_auc": eval_last(["prediction_auc", "auc"], 0.5),
            "test_return_mae": eval_last(["return_pred_mae", "mae"], 0.0),
            "test_return_mse": eval_last(["return_pred_mse", "mse"], 0.0),
        }
        final_metrics["return_rmse"] = self._safe_float(np.sqrt(max(final_metrics["return_mse"], 0.0)), 0.0)
        final_metrics["val_return_rmse"] = self._safe_float(
            np.sqrt(max(final_metrics["val_return_mse"], 0.0)),
            0.0,
        )
        final_metrics["test_return_rmse"] = self._safe_float(
            np.sqrt(max(final_metrics["test_return_mse"], 0.0)),
            0.0,
        )

        comparison = self._compare_with_baseline(final_metrics)
        should_replace = self._should_replace_model(comparison)

        model_path = self._keras_model_path() if should_replace else self._keras_candidate_path()
        model.save(model_path)

        if should_replace:
            self.model = model
        production_available = bool(should_replace or self.model is not None)

        metadata = {
            "model_name": self.model_name,
            "available": production_available,
            "created_at": datetime.now().isoformat(),
            "backend": "tensorflow",
            "model_tier": model_tier,
            "training_samples": total_samples,
            "data_split": {
                "strategy": "chronological",
                "train_samples": int(train_size),
                "val_samples": int(val_size),
                "test_samples": int(total_samples - val_end),
            },
            "epochs_trained": int(len(history.history.get("loss", []))),
            "final_metrics": final_metrics,
            "feature_dims": {
                "news": self.news_feature_dim,
                "sector": self.sector_feature_dim,
                "tech": self.tech_feature_dim,
                "relevance": self.relevance_dim,
                "text": self.text_feature_dim,
            },
            "text_encoder": {
                "type": "native_hashing" if self.text_hash_dim > 0 else "none",
                "stat_dim": self.text_stat_dim,
                "hash_dim": self.text_hash_dim,
            },
            "targets": {
                "classification": "label_up_5d",
                "regression": "future_ret_5d(decimal)",
                "regression_clip": [-0.20, 0.20],
            },
            "baseline_comparison": comparison,
            "production_model_replaced": should_replace,
            "model_path": model_path,
            "candidate_only": not should_replace,
            "tensorflow_available": TENSORFLOW_AVAILABLE,
        }

        self._save_metadata(metadata)

        message = "模型训练完成并已替换线上模型"
        if not should_replace:
            message = "模型训练完成，但未优于基线，仅保存为候选模型"

        return {
            "success": True,
            "message": message,
            "model_path": model_path,
            "metadata": metadata,
        }

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info: Dict[str, Any] = {}

        metadata_path = self._metadata_path()
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as file_obj:
                    info = json.load(file_obj)
            except Exception as e:
                logger.warning(f"读取模型信息失败: {e}")

        info.setdefault("model_name", self.model_name)
        info["tensorflow_available"] = TENSORFLOW_AVAILABLE

        runtime_backend = "rule_based"
        if self.model is not None:
            runtime_backend = "tensorflow"
        info["runtime_backend"] = runtime_backend

        info["available"] = bool(self.model is not None)
        return info

    def train(
        self,
        training_data: Dict,
        epochs: int = 50,
        batch_size: int = 32,
        model_tier: str = "heavy",
    ) -> Dict:
        """
        训练多模态模型。

        Args:
            training_data: 训练数据字典，包含：
                - news_features: (n_samples, news_feature_dim)
                - sector_features: (n_samples, sector_feature_dim)
                - tech_features: (n_samples, tech_feature_dim)
                - relevance_features: (n_samples, relevance_dim)
                - text_features: (n_samples, text_feature_dim)，可选
                - labels: (n_samples,)
                - return_targets: (n_samples,) 未来收益率（十进制），可选
            epochs: 训练轮数
            batch_size: 批量大小
            model_tier: 兼容字段，当前固定 heavy

        Returns:
            训练结果
        """
        tier = "heavy"

        try:
            arrays = self._prepare_training_arrays(training_data)

            if TENSORFLOW_AVAILABLE:
                return self._train_tensorflow(
                    arrays=arrays,
                    epochs=epochs,
                    batch_size=batch_size,
                    model_tier=tier,
                )

            return {
                "success": False,
                "message": "TensorFlow 不可用，无法训练多模态模型",
            }

        except Exception as e:
            logger.error(f"训练模型失败: {e}")
            import traceback

            traceback.print_exc()
            return {
                "success": False,
                "message": str(e),
            }

    def encode_news_features(self, news_list: List[Dict]) -> np.ndarray:
        """编码新闻统计特征向量"""
        features = np.zeros(self.news_feature_dim)

        if not news_list:
            return features.reshape(1, -1)

        sentiments = [news.get("sentiment", 0) for news in news_list]
        importances = [news.get("importance", 0.5) for news in news_list]

        # 基础统计
        features[0] = np.mean(sentiments)
        features[1] = np.std(sentiments) if len(sentiments) > 1 else 0
        features[2] = np.mean(importances)
        features[3] = min(len(news_list) / 50, 1)

        # 情感分布
        features[4] = sum(1 for val in sentiments if val > 0.2) / max(len(sentiments), 1)
        features[5] = sum(1 for val in sentiments if val < -0.2) / max(len(sentiments), 1)

        # 时间加权情感
        current_time = datetime.now()
        time_weights = []
        for news in news_list:
            publish_time = news.get("publish_time", "")
            if publish_time:
                try:
                    dt_value = datetime.strptime(publish_time, "%Y-%m-%d %H:%M:%S")
                    hours_ago = (current_time - dt_value).total_seconds() / 3600
                    weight = np.exp(-hours_ago / 24)
                except Exception:
                    weight = 0.5
            else:
                weight = 0.5
            time_weights.append(weight)

        weighted_sentiment = np.average(sentiments, weights=time_weights) if time_weights else 0
        features[6] = weighted_sentiment

        # 类别分布
        categories = {}
        for news in news_list:
            for category in news.get("categories", []):
                categories[category] = categories.get(category, 0) + 1

        for index, category in enumerate(list(categories.keys())[:20]):
            features[7 + index] = categories[category] / len(news_list)

        return features.reshape(1, -1)

    def encode_news_text_features(self, news_list: List[Dict]) -> np.ndarray:
        """编码新闻文本向量（统计特征 + Hashing 向量）"""
        features = np.zeros(self.text_feature_dim)
        if not news_list:
            return features.reshape(1, -1)

        texts: List[str] = []
        title_lengths = []
        content_lengths = []
        pos_hits = 0
        neg_hits = 0
        punct_count = 0
        unique_chars = set()

        for news in news_list:
            title = str(news.get("title", ""))
            content = str(news.get("content", ""))
            merged_text = f"{title} {content}".strip()

            if merged_text:
                texts.append(merged_text)
                unique_chars.update(set(merged_text))

            title_lengths.append(len(title))
            content_lengths.append(len(content))

            for keyword in self.positive_keywords:
                pos_hits += merged_text.count(keyword)
            for keyword in self.negative_keywords:
                neg_hits += merged_text.count(keyword)

            punct_count += merged_text.count("!") + merged_text.count("！")
            punct_count += merged_text.count("?") + merged_text.count("？")

        total_hits = max(pos_hits + neg_hits, 1)
        features[0] = min(self._safe_float(np.mean(title_lengths), 0.0) / 40.0, 1.0)
        features[1] = min(self._safe_float(np.mean(content_lengths), 0.0) / 300.0, 1.0)
        features[2] = self._safe_float(pos_hits / total_hits, 0.0)
        features[3] = self._safe_float(neg_hits / total_hits, 0.0)
        features[4] = min(self._safe_float(punct_count / max(len(texts), 1), 0.0) / 8.0, 1.0)
        features[5] = min(self._safe_float(len(unique_chars), 0.0) / 800.0, 1.0)
        features[6] = min(self._safe_float(len(texts), 0.0) / 20.0, 1.0)

        sentiments = [self._safe_float(news.get("sentiment", 0.0), 0.0) for news in news_list]
        features[7] = self._safe_float(np.mean(sentiments), 0.0) if sentiments else 0.0

        # 文本哈希向量拼接到后半段（原生实现，避免依赖 sklearn）
        if texts and self.text_hash_dim > 0:
            try:
                hashed_dense = self._encode_text_hash_features(texts)
                used_dim = min(len(hashed_dense), self.text_hash_dim)
                start = self.text_stat_dim
                features[start : start + used_dim] = hashed_dense[:used_dim]
            except Exception as e:
                logger.warning(f"文本哈希向量编码失败: {e}")

        return features.reshape(1, -1)

    def _encode_text_hash_features(self, texts: List[str]) -> np.ndarray:
        """将文本映射到固定维度哈希向量（char n-gram）。"""
        vector = np.zeros(self.text_hash_dim, dtype=float)
        if not texts or self.text_hash_dim <= 0:
            return vector

        total = 0
        for raw_text in texts:
            text = str(raw_text or "").strip()
            if not text:
                continue

            length = len(text)
            for ngram in (2, 3, 4):
                if length < ngram:
                    continue

                for idx in range(length - ngram + 1):
                    token = text[idx : idx + ngram]
                    token_hash = hashlib.md5(token.encode("utf-8")).hexdigest()
                    position = int(token_hash[:8], 16) % self.text_hash_dim
                    vector[position] += 1.0
                    total += 1

        if total > 0:
            vector /= float(total)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm

        return vector

    def encode_sector_features(self, sector_impact: Dict[str, float]) -> np.ndarray:
        """编码领域影响特征向量"""
        features = np.zeros(self.sector_feature_dim)

        for index, sector in enumerate(self.sectors[: self.sector_feature_dim]):
            features[index] = sector_impact.get(sector, 0)

        return features.reshape(1, -1)

    def encode_technical_features(self, df: pd.DataFrame) -> np.ndarray:
        """编码技术指标特征"""
        features = np.zeros(self.tech_feature_dim)

        if df.empty or len(df) < 20:
            return features.reshape(1, -1)

        try:
            close = df["close"]

            # 辅助函数：安全获取值，避免 NaN
            def safe_get(val, default=0.0):
                try:
                    float_val = float(val)
                    return float_val if not (np.isnan(float_val) or np.isinf(float_val)) else default
                except Exception:
                    return default

            # 收益率
            features[0] = safe_get((close.iloc[-1] / close.iloc[-5] - 1) * 100)
            features[1] = safe_get((close.iloc[-1] / close.iloc[-20] - 1) * 100)

            # 波动率
            returns = close.pct_change()
            std5 = returns.tail(5).std()
            std20 = returns.tail(20).std()
            features[2] = safe_get(std5 * 100) if not np.isnan(std5) else 0
            features[3] = safe_get(std20 * 100) if not np.isnan(std20) else 0

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1]
            features[4] = safe_get(rsi_val / 100) if not np.isnan(rsi_val) else 0.5

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_val = macd.iloc[-1]
            features[5] = safe_get(macd_val / close.iloc[-1] * 100)

            # 布林带位置
            ma20 = close.rolling(20).mean()
            std20_bb = close.rolling(20).std()
            upper = ma20 + 2 * std20_bb
            lower = ma20 - 2 * std20_bb
            bb_position = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-10)
            features[6] = safe_get(bb_position, 0.5)

            # 成交量比率
            volume = df["volume"]
            vol_mean = volume.tail(20).mean()
            if vol_mean > 0:
                volume_ratio = volume.iloc[-1] / vol_mean
                features[7] = min(safe_get(volume_ratio / 5, 0.2), 1)

            # 价格位置（使用可用数据范围）
            lookback = min(60, len(close))
            rolling_high = close.rolling(lookback).max()
            rolling_low = close.rolling(lookback).min()
            high_val = rolling_high.iloc[-1]
            low_val = rolling_low.iloc[-1]
            if high_val > low_val:
                price_position = (close.iloc[-1] - low_val) / (high_val - low_val)
                features[8] = safe_get(price_position, 0.5)
            else:
                features[8] = 0.5

            # 均线趋势
            ma5 = close.rolling(5).mean()
            ma10 = close.rolling(10).mean()
            ma20_val = close.rolling(20).mean()
            if len(df) >= 20:
                features[9] = 1 if ma5.iloc[-1] > ma10.iloc[-1] > ma20_val.iloc[-1] else -1
                features[10] = safe_get((ma5.iloc[-1] - ma20_val.iloc[-1]) / ma20_val.iloc[-1] * 100)

            # 动量
            momentum = close.pct_change(10).iloc[-1] * 100
            features[11] = safe_get(momentum)

        except Exception as e:
            logger.warning(f"计算技术指标失败: {e}")

        return features.reshape(1, -1)

    def encode_relevance_features(self, relevance_matrix: np.ndarray, stock_idx: int) -> np.ndarray:
        """编码相关性特征"""
        features = np.zeros(self.relevance_dim)

        if relevance_matrix is None or relevance_matrix.size == 0:
            return features.reshape(1, -1)

        try:
            correlations = relevance_matrix[stock_idx]

            features[0] = np.mean(correlations)
            features[1] = np.max(correlations)
            features[2] = np.std(correlations)
            features[3] = np.sum(correlations > 0.5) / len(correlations)
            features[4] = np.sum(correlations > 0.7) / len(correlations)

            sorted_corr = np.sort(correlations)[::-1]
            for index in range(min(10, len(sorted_corr))):
                features[5 + index] = sorted_corr[index]

        except Exception as e:
            logger.warning(f"编码相关性特征失败: {e}")

        return features.reshape(1, -1)

    def predict(
        self,
        news_features: np.ndarray,
        sector_features: np.ndarray,
        tech_features: np.ndarray,
        relevance_features: np.ndarray,
        text_features: Optional[np.ndarray] = None,
    ) -> Dict:
        """执行预测"""
        news_features = self._normalize_2d(news_features, self.news_feature_dim)
        sector_features = self._normalize_2d(sector_features, self.sector_feature_dim)
        tech_features = self._normalize_2d(tech_features, self.tech_feature_dim)
        relevance_features = self._normalize_2d(relevance_features, self.relevance_dim)

        if text_features is None:
            text_features = np.zeros((news_features.shape[0], self.text_feature_dim), dtype=float)
        text_features = self._normalize_2d(text_features, self.text_feature_dim)

        predicted_return_decimal: Optional[float] = None
        if TENSORFLOW_AVAILABLE and self.model is not None:
            input_count = len(getattr(self.model, "inputs", []))
            if input_count >= 5:
                model_inputs = [
                    news_features,
                    sector_features,
                    tech_features,
                    relevance_features,
                    text_features,
                ]
            else:
                # 兼容旧模型（4路输入）
                model_inputs = [news_features, sector_features, tech_features, relevance_features]

            model_output = self.model.predict(model_inputs, verbose=0)
            if isinstance(model_output, (list, tuple)):
                prob = float(np.asarray(model_output[0]).reshape(-1)[0])
                if len(model_output) > 1:
                    predicted_return_decimal = float(np.asarray(model_output[1]).reshape(-1)[0])
            else:
                prob = float(np.asarray(model_output).reshape(-1)[0])
            backend = "tensorflow"
        else:
            prob, predicted_return_decimal = self._simple_predict(
                news_features,
                sector_features,
                tech_features,
                relevance_features,
                text_features,
            )
            backend = "rule_based"

        if np.isnan(prob) or np.isinf(prob):
            prob = 0.5

        if predicted_return_decimal is not None:
            if np.isnan(predicted_return_decimal) or np.isinf(predicted_return_decimal):
                predicted_return_decimal = None
            else:
                predicted_return_decimal = float(np.clip(predicted_return_decimal, -0.20, 0.20))

        if predicted_return_decimal is not None:
            prediction = 1 if predicted_return_decimal >= 0 else 0
        else:
            prediction = 1 if prob > 0.5 else 0
        confidence = prob if prediction == 1 else 1 - prob

        return {
            "success": True,
            "prediction": int(prediction),
            "prediction_text": "上涨" if prediction == 1 else "下跌",
            "probability": round(float(prob), 4),
            "confidence": round(float(confidence), 4),
            "predicted_return_decimal": (
                round(float(predicted_return_decimal), 6)
                if predicted_return_decimal is not None
                else None
            ),
            "model_type": "multimodal",
            "backend": backend,
            "features_used": {
                "news": bool(np.any(news_features)),
                "sector": bool(np.any(sector_features)),
                "tech": bool(np.any(tech_features)),
                "relevance": bool(np.any(relevance_features)),
                "text": bool(np.any(text_features)),
            },
        }

    def _simple_predict(
        self,
        news_features: np.ndarray,
        sector_features: np.ndarray,
        tech_features: np.ndarray,
        relevance_features: np.ndarray,
        text_features: np.ndarray,
    ) -> Tuple[float, float]:
        """规则版预测（无训练模型时使用）"""
        # 新闻情感权重 (-1 到 1)
        news_score = float(news_features[0, 0]) * 0.5 + float(news_features[0, 6]) * 0.5

        # 领域影响权重
        sector_mean = np.mean(sector_features)
        sector_score = float(sector_mean) if not np.isnan(sector_mean) else 0.0

        # 技术指标评分 (-0.5 到 0.5)
        tech_score = 0.0
        try:
            ma_trend = float(tech_features[0, 9])
            tech_score += ma_trend * 0.15

            rsi = float(tech_features[0, 4])
            if rsi < 0.3:
                tech_score += 0.1
            elif rsi > 0.7:
                tech_score -= 0.1

            bb_pos = float(tech_features[0, 6])
            if bb_pos < 0.2:
                tech_score += 0.1
            elif bb_pos > 0.8:
                tech_score -= 0.1

            ret_5d = float(tech_features[0, 0])
            if ret_5d > 2:
                tech_score -= 0.1
            elif ret_5d < -2:
                tech_score += 0.1

            momentum = float(tech_features[0, 11])
            if momentum < -3:
                tech_score -= 0.1
            elif momentum > 3:
                tech_score += 0.05
        except (IndexError, ValueError):
            pass

        # 相关性权重
        relevance_score = 0.0
        try:
            rel_mean = float(relevance_features[0, 0])
            if rel_mean > 0.6:
                relevance_score += 0.1
            elif rel_mean < 0.3:
                relevance_score -= 0.05
        except (IndexError, ValueError):
            pass

        # 文本权重（Phase 4）
        text_score = 0.0
        try:
            polarity = float(text_features[0, 2]) - float(text_features[0, 3])
            sentiment_hint = float(text_features[0, 7])
            text_score += polarity * 0.12 + sentiment_hint * 0.08
        except (IndexError, ValueError):
            pass

        # 加权融合
        combined = (
            news_score * 0.28
            + sector_score * 0.18
            + tech_score * 0.36
            + relevance_score * 0.10
            + text_score * 0.08
        )

        if np.isnan(combined) or np.isinf(combined):
            combined = 0.0

        prob = 1 / (1 + np.exp(-combined * 8))
        expected_return_decimal = float(np.tanh(combined) * 0.08)
        return float(prob), expected_return_decimal

    def predict_stock(
        self,
        stock_code: str,
        kline_df: pd.DataFrame,
        news_list: List[Dict] = None,
        sector_impact: Dict[str, float] = None,
        relevance_matrix: np.ndarray = None,
        stock_idx: int = 0,
    ) -> Dict:
        """
        单股预测。

        Args:
            stock_code: 股票代码
            kline_df: K线数据
            news_list: 新闻列表
            sector_impact: 领域影响
            relevance_matrix: 相关性矩阵
            stock_idx: 股票索引
        """
        if news_list is None:
            news_list = []
        if sector_impact is None:
            sector_impact = {}

        # 编码特征
        news_features = self.encode_news_features(news_list)
        text_features = self.encode_news_text_features(news_list)
        sector_features = self.encode_sector_features(sector_impact)
        tech_features = self.encode_technical_features(kline_df)

        if relevance_matrix is not None:
            relevance_features = self.encode_relevance_features(relevance_matrix, stock_idx)
        else:
            relevance_features = np.zeros((1, self.relevance_dim))

        # 预测
        prediction = self.predict(
            news_features,
            sector_features,
            tech_features,
            relevance_features,
            text_features,
        )

        latest_price = float(kline_df["close"].iloc[-1]) if not kline_df.empty else 0.0

        def safe_float(val, default=0.0):
            try:
                float_val = float(val)
                return float_val if not (np.isnan(float_val) or np.isinf(float_val)) else default
            except Exception:
                return default

        news_sentiment = safe_float(news_features[0, 0])
        news_weighted = safe_float(news_features[0, 6])
        text_polarity = safe_float(text_features[0, 2] - text_features[0, 3])
        tech_trend_val = safe_float(tech_features[0, 9])
        rsi_val = safe_float(tech_features[0, 4])

        sector_mean = 0.0
        if sector_impact:
            values = [safe_float(val) for val in sector_impact.values()]
            valid_values = [val for val in values if val != 0.0]
            if valid_values:
                sector_mean = np.mean(valid_values)

        prediction.update(
            {
                "stock_code": stock_code,
                "latest_price": round(latest_price, 2),
                "news_count": len(news_list),
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "feature_summary": {
                    "news_sentiment": round(news_sentiment, 4),
                    "news_weighted_sentiment": round(news_weighted, 4),
                    "text_polarity": round(text_polarity, 4),
                    "sector_impact": round(float(sector_mean), 4),
                    "tech_trend": "up" if tech_trend_val > 0 else "down",
                    "rsi": round(rsi_val, 4),
                },
            }
        )

        expected_return_source = "regression_head"
        predicted_return_decimal = prediction.get("predicted_return_decimal")
        if predicted_return_decimal is not None:
            expected_return = float(predicted_return_decimal) * 100.0
        else:
            # 兼容旧模型：若无回归头输出，则按概率与波动率估算收益幅度
            prob = safe_float(prediction.get("probability"), 0.5)
            vol_20d_pct = abs(safe_float(tech_features[0, 3], 0.0))
            # vol_20d_pct 来自 20 日波动率（百分比），用于动态收益尺度
            scale_pct = float(np.clip(vol_20d_pct * 1.8, 0.8, 12.0))
            direction_strength = float(np.tanh((prob - 0.5) * 3.8))
            expected_return = direction_strength * scale_pct
            expected_return_source = "probability_volatility_fallback"

        expected_return = float(np.clip(expected_return, -20.0, 20.0))
        predicted_price = latest_price * (1 + expected_return / 100.0)

        prediction["expected_return"] = round(expected_return, 2)
        prediction["predicted_price"] = round(predicted_price, 2)
        prediction["expected_return_source"] = expected_return_source
        prediction["success"] = True

        return prediction

    @staticmethod
    def _safe_float(value, default: Optional[float] = 0.0) -> Optional[float]:
        """安全转换为 float。default 可为 None。"""
        try:
            result = float(value)
        except Exception:
            return default

        if np.isnan(result) or np.isinf(result):
            return default
        return result


# 单例获取函数
def get_multimodal_predictor() -> MultiModalPredictor:
    """获取多模态预测器单例"""
    return MultiModalPredictor()
