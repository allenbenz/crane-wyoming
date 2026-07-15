// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Andreas Schneider <asn@cryptomilk.org>
//
// Based on the `engine` module of Crane's `crane-serve` crate
// (https://github.com/lucasjinreal/Crane), Copyright (c) 2024 Nicholas Jela,
// licensed under the MIT License.

//! TTS model factory for automatic model type detection and construction.
//!
//! Supports auto-detection from `config.json`'s `model_type` / `architectures`
//! fields (Qwen3-TTS) or `params.json`'s `model_type` field (Voxtral-TTS), or
//! explicit model type specification via CLI.

use anyhow::Result;
use candle_core::{DType, Device};
use serde::Deserialize;
use std::path::Path;

// ─────────────────────────────────────────────────────────────
//  Enums
// ─────────────────────────────────────────────────────────────

/// Supported TTS model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Detect the model type from the model directory's config files.
    Auto,
    /// Qwen3-TTS.
    Qwen3TTS,
    /// Voxtral-4B-TTS.
    VoxtralTTS,
}

impl ModelType {
    // Infallible convenience constructor, not the fallible std::str::FromStr
    // trait (unknown strings fall back to `Auto` instead of erroring).
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "qwen3_tts" | "qwen3tts" | "qwen3-tts" | "tts" => Self::Qwen3TTS,
            "voxtral_tts" | "voxtral-tts" | "voxtral" | "voxtral_4b" => Self::VoxtralTTS,
            _ => Self::Auto,
        }
    }

    /// Returns the display name used in logs and registration.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Qwen3TTS => "qwen3_tts",
            Self::VoxtralTTS => "voxtral_tts",
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Detection
// ─────────────────────────────────────────────────────────────

/// Minimal subset of `HuggingFace` `config.json` for architecture detection.
#[derive(Deserialize, Default)]
struct HfConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
}

/// Minimal subset of Mistral `params.json` for architecture detection.
#[derive(Deserialize, Default)]
struct MistralConfig {
    model_type: Option<String>,
}

/// Auto-detect the TTS model type from `config.json`/`params.json` in the
/// model directory, falling back to a path-name heuristic.
#[must_use]
pub fn detect_model_type(model_path: &str) -> ModelType {
    let path = Path::new(model_path);

    let config_path = if path.is_file() {
        path.parent().map(|p| p.join("config.json"))
    } else {
        Some(path.join("config.json"))
    };

    if let Some(config_path) = config_path
        && let Ok(data) = std::fs::read(&config_path)
        && let Ok(config) = serde_json::from_slice::<HfConfig>(&data)
    {
        if let Some(ref mt) = config.model_type
            && matches!(mt.to_lowercase().as_str(), "qwen3_tts" | "qwen3tts")
        {
            return ModelType::Qwen3TTS;
        }
        if let Some(ref archs) = config.architectures {
            for arch in archs {
                let a = arch.to_lowercase();
                if a.contains("qwen3ttsforconditional") || a.contains("qwen3_tts") {
                    return ModelType::Qwen3TTS;
                }
            }
        }
    }

    let params_path = if path.is_file() {
        path.parent().map(|p| p.join("params.json"))
    } else {
        Some(path.join("params.json"))
    };
    if let Some(params_path) = params_path
        && let Ok(data) = std::fs::read(&params_path)
        && let Ok(config) = serde_json::from_slice::<MistralConfig>(&data)
        && let Some(ref mt) = config.model_type
        && mt == "voxtral_tts"
    {
        return ModelType::VoxtralTTS;
    }

    let path_lower = model_path.to_lowercase();
    if path_lower.contains("voxtral") {
        ModelType::VoxtralTTS
    } else {
        tracing::warn!(
            "Could not auto-detect TTS model type from '{model_path}', defaulting to Qwen3-TTS"
        );
        ModelType::Qwen3TTS
    }
}

// ─────────────────────────────────────────────────────────────
//  Factory
// ─────────────────────────────────────────────────────────────

/// Resolve `ModelType::Auto` to a concrete type.
#[must_use]
pub fn resolve(model_type: ModelType, model_path: &str) -> ModelType {
    if model_type == ModelType::Auto {
        detect_model_type(model_path)
    } else {
        model_type
    }
}

/// Create a TTS model as a trait object.
///
/// Dispatches on [`ModelType`] and returns a `Box<dyn Tts + Send>`, allowing
/// callers (e.g. [`crate::engine::runtime::ModelRuntime`]) to work with any
/// TTS model through the [`crane::audio::tts::Tts`] trait without depending
/// on concrete model types.
///
/// # Errors
///
/// Returns an error if `model_type` is `Auto` (must be resolved first) or if
/// the model fails to load from `model_path`.
pub fn create_tts(
    model_type: ModelType,
    model_path: &str,
    device: &Device,
    dtype: &DType,
) -> Result<Box<dyn crane::audio::tts::Tts + Send>> {
    tracing::info!("Creating TTS model: {:?}", model_type);

    match model_type {
        ModelType::Qwen3TTS => Ok(Box::new(crane_core::models::qwen3_tts::Model::new(
            model_path, device, dtype,
        )?)),
        ModelType::VoxtralTTS => Ok(Box::new(crane_core::models::voxtral_tts::Model::new(
            model_path, device, dtype,
        )?)),
        ModelType::Auto => anyhow::bail!("ModelType::Auto must be resolved before create_tts()"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ModelType::from_str ──

    #[test]
    fn model_type_from_str_qwen3_variants() {
        assert_eq!(ModelType::from_str("qwen3_tts"), ModelType::Qwen3TTS);
        assert_eq!(ModelType::from_str("qwen3tts"), ModelType::Qwen3TTS);
        assert_eq!(ModelType::from_str("qwen3-tts"), ModelType::Qwen3TTS);
        assert_eq!(ModelType::from_str("QWEN3_TTS"), ModelType::Qwen3TTS);
    }

    #[test]
    fn model_type_from_str_voxtral_variants() {
        assert_eq!(ModelType::from_str("voxtral_tts"), ModelType::VoxtralTTS);
        assert_eq!(ModelType::from_str("voxtral-tts"), ModelType::VoxtralTTS);
        assert_eq!(ModelType::from_str("voxtral"), ModelType::VoxtralTTS);
        assert_eq!(ModelType::from_str("voxtral_4b"), ModelType::VoxtralTTS);
        assert_eq!(ModelType::from_str("VOXTRAL"), ModelType::VoxtralTTS);
    }

    #[test]
    fn model_type_from_str_auto_fallback() {
        assert_eq!(ModelType::from_str("auto"), ModelType::Auto);
        assert_eq!(ModelType::from_str("unknown"), ModelType::Auto);
        assert_eq!(ModelType::from_str(""), ModelType::Auto);
    }

    #[test]
    fn model_type_display_name() {
        assert_eq!(ModelType::Auto.display_name(), "auto");
        assert_eq!(ModelType::Qwen3TTS.display_name(), "qwen3_tts");
        assert_eq!(ModelType::VoxtralTTS.display_name(), "voxtral_tts");
    }

    // ── detect_model_type with temp files ──

    #[test]
    fn detect_from_config_json_model_type_qwen3_tts() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"model_type": "qwen3_tts"}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen3TTS);
    }

    #[test]
    fn detect_from_config_json_architectures_qwen3_tts() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(
            &config,
            r#"{"architectures": ["Qwen3TTSForConditionalGeneration"]}"#,
        )
        .unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen3TTS);
    }

    #[test]
    fn detect_from_params_json_voxtral() {
        let dir = tempfile::tempdir().unwrap();
        let params = dir.path().join("params.json");
        std::fs::write(&params, r#"{"model_type": "voxtral_tts"}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::VoxtralTTS);
    }

    #[test]
    fn detect_path_heuristic_voxtral() {
        let result = detect_model_type("/models/Voxtral-4B-TTS-2603");
        assert_eq!(result, ModelType::VoxtralTTS);
    }

    #[test]
    fn detect_fallback_unknown_defaults_to_qwen3_tts() {
        let dir = tempfile::tempdir().unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen3TTS);
    }

    // ── resolve ──

    #[test]
    fn resolve_auto_delegates_to_detect() {
        let result = resolve(ModelType::Auto, "/models/Voxtral-4B-TTS-2603");
        assert_eq!(result, ModelType::VoxtralTTS);
    }

    #[test]
    fn resolve_explicit_type_is_passthrough() {
        let result = resolve(ModelType::Qwen3TTS, "/models/whatever");
        assert_eq!(result, ModelType::Qwen3TTS);
    }

    // ── create_tts ──

    #[test]
    fn create_tts_rejects_auto() {
        let result = create_tts(
            ModelType::Auto,
            "/models/whatever",
            &Device::Cpu,
            &DType::F32,
        );
        let Err(err) = result else {
            panic!("expected create_tts to reject ModelType::Auto");
        };
        assert!(err.to_string().contains("must be resolved"));
    }
}
