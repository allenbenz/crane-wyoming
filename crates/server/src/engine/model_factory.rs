// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Andreas Schneider <asn@cryptomilk.org>
//
// Based on the `engine` module of Crane's `crane-serve` crate
// (https://github.com/lucasjinreal/Crane), Copyright (c) 2024 Nicholas Jela,
// licensed under the MIT License.

//! TTS and ASR model factory for automatic model type detection and
//! construction.
//!
//! Supports auto-detection from `config.json`'s `model_type` / `architectures`
//! fields (Qwen3-TTS, Qwen3-ASR) or `params.json`'s `model_type` field
//! (Voxtral-TTS), or explicit model type specification via CLI.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use serde::Deserialize;
use std::path::{Path, PathBuf};

// ─────────────────────────────────────────────────────────────
//  Enums
// ─────────────────────────────────────────────────────────────

/// Supported TTS/ASR model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Detect the model type from the model directory's config files.
    Auto,
    /// Qwen3-TTS.
    Qwen3TTS,
    /// Voxtral-4B-TTS.
    VoxtralTTS,
    /// Qwen3-ASR.
    Qwen3ASR,
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
            "qwen3_asr" | "qwen3asr" | "qwen3-asr" | "asr" => Self::Qwen3ASR,
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
            Self::Qwen3ASR => "qwen3_asr",
        }
    }

    /// Returns `true` if this is a TTS model type.
    ///
    /// `Auto` is neither a TTS nor an ASR type (it must be resolved to a
    /// concrete type first), so this returns `false` for it.
    #[must_use]
    pub fn is_tts(&self) -> bool {
        match self {
            Self::Qwen3TTS | Self::VoxtralTTS => true,
            Self::Qwen3ASR | Self::Auto => false,
        }
    }

    /// Returns `true` if this is an ASR model type.
    ///
    /// `Auto` is neither a TTS nor an ASR type (it must be resolved to a
    /// concrete type first), so this returns `false` for it.
    #[must_use]
    pub fn is_asr(&self) -> bool {
        match self {
            Self::Qwen3ASR => true,
            Self::Qwen3TTS | Self::VoxtralTTS | Self::Auto => false,
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

/// Probe whether `dir` contains a recognized TTS or ASR model, returning
/// its type if so.
///
/// Only looks at `config.json`/`params.json`; unlike [`detect_model_type`],
/// it never falls back to a path-name heuristic, so it returns `None` for
/// directories that don't contain a recognizable model. This makes it
/// suitable for scanning a parent directory (see [`discover_models`] and
/// [`discover_asr_models`]), where non-model subdirectories must be
/// silently skipped rather than misidentified.
#[must_use]
pub fn probe_model_type(dir: &Path) -> Option<ModelType> {
    let config_path = dir.join("config.json");
    if let Ok(data) = std::fs::read(&config_path)
        && let Ok(config) = serde_json::from_slice::<HfConfig>(&data)
    {
        if let Some(ref mt) = config.model_type {
            match mt.to_lowercase().as_str() {
                "qwen3_tts" | "qwen3tts" => return Some(ModelType::Qwen3TTS),
                "qwen3_asr" | "qwen3asr" => return Some(ModelType::Qwen3ASR),
                _ => {},
            }
        }
        if let Some(ref archs) = config.architectures {
            for arch in archs {
                let a = arch.to_lowercase();
                if a.contains("qwen3ttsforconditional") || a.contains("qwen3_tts") {
                    return Some(ModelType::Qwen3TTS);
                } else if a.contains("qwen3asrforconditional") || a.contains("qwen3_asr") {
                    return Some(ModelType::Qwen3ASR);
                }
            }
        }
    }

    let params_path = dir.join("params.json");
    if let Ok(data) = std::fs::read(&params_path)
        && let Ok(config) = serde_json::from_slice::<MistralConfig>(&data)
        && let Some(ref mt) = config.model_type
        && mt == "voxtral_tts"
    {
        return Some(ModelType::VoxtralTTS);
    }

    None
}

/// Auto-detect the TTS/ASR model type from `config.json`/`params.json` in
/// the model directory, falling back to a path-name heuristic.
#[must_use]
pub fn detect_model_type(model_path: &str) -> ModelType {
    let path = Path::new(model_path);
    let dir = if path.is_file() {
        path.parent().unwrap_or(path)
    } else {
        path
    };

    if let Some(model_type) = probe_model_type(dir) {
        return model_type;
    }

    let path_lower = model_path.to_lowercase();
    if path_lower.contains("voxtral") {
        ModelType::VoxtralTTS
    } else if path_lower.contains("qwen3-asr") || path_lower.contains("qwen3_asr") {
        ModelType::Qwen3ASR
    } else {
        tracing::warn!(
            "Could not auto-detect model type from '{model_path}', defaulting to Qwen3-TTS"
        );
        ModelType::Qwen3TTS
    }
}

/// A TTS or ASR model discovered by [`discover_models`]/[`discover_asr_models`].
#[derive(Debug, PartialEq, Eq)]
pub struct DiscoveredModel {
    /// Full path to the model directory.
    pub path: PathBuf,
    /// Directory name (the final path component), used as the model's
    /// registration name.
    pub name: String,
    /// Detected model architecture.
    pub model_type: ModelType,
}

/// Scan `parent_dir` for immediate subdirectories containing recognized
/// TTS models.
///
/// Each subdirectory is probed with [`probe_model_type`]; subdirectories
/// that don't contain a recognized model (no `config.json`/`params.json`
/// with a known `model_type`) are silently skipped. Results are sorted
/// alphabetically by directory name, giving deterministic default model
/// ordering.
///
/// # Errors
///
/// Returns an error if `parent_dir` cannot be read.
pub fn discover_models(parent_dir: &Path) -> Result<Vec<DiscoveredModel>> {
    let entries = std::fs::read_dir(parent_dir).map_err(|e| {
        anyhow::anyhow!(
            "cannot read model directory '{}': {e}",
            parent_dir.display()
        )
    })?;

    let mut models = Vec::new();
    for entry in entries {
        let path = entry
            .with_context(|| format!("cannot read entry in '{}'", parent_dir.display()))?
            .path();
        match path.metadata() {
            Ok(meta) if meta.is_dir() => {},
            Ok(_) => continue, // not a directory: silently skip, e.g. stray files
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "cannot stat entry; skipping");
                continue;
            },
        }
        if let Some(model_type) = probe_model_type(&path)
            && model_type.is_tts()
        {
            let name = path
                .file_name()
                .map_or_else(|| "tts".to_string(), |n| n.to_string_lossy().into_owned());
            models.push(DiscoveredModel {
                path,
                name,
                model_type,
            });
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

/// Scan `parent_dir` for immediate subdirectories containing recognized
/// ASR models.
///
/// Identical to [`discover_models`] except it keeps only ASR model types
/// (see [`ModelType::is_asr`]), so pointing this and [`discover_models`] at
/// the same parent directory (e.g. `<model-path>/tts` and
/// `<model-path>/asr`) won't cross-load a TTS model as ASR or vice versa.
///
/// # Errors
///
/// Returns an error if `parent_dir` cannot be read.
pub fn discover_asr_models(parent_dir: &Path) -> Result<Vec<DiscoveredModel>> {
    let entries = std::fs::read_dir(parent_dir).map_err(|e| {
        anyhow::anyhow!(
            "cannot read model directory '{}': {e}",
            parent_dir.display()
        )
    })?;

    let mut models = Vec::new();
    for entry in entries {
        let path = entry
            .with_context(|| format!("cannot read entry in '{}'", parent_dir.display()))?
            .path();
        match path.metadata() {
            Ok(meta) if meta.is_dir() => {},
            Ok(_) => continue, // not a directory: silently skip, e.g. stray files
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "cannot stat entry; skipping");
                continue;
            },
        }
        if let Some(model_type) = probe_model_type(&path)
            && model_type.is_asr()
        {
            let name = path
                .file_name()
                .map_or_else(|| "asr".to_string(), |n| n.to_string_lossy().into_owned());
            models.push(DiscoveredModel {
                path,
                name,
                model_type,
            });
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

/// Shared implementation behind [`resolve_models_to_load`] and
/// [`resolve_asr_models_to_load`]: resolve `requested` names against
/// `discovered` models, reporting errors against `flag_name` (e.g.
/// `--model-tts`).
///
/// If `requested` is empty, every model in `discovered` is returned (in its
/// existing, alphabetical order). Otherwise, exactly the named models are
/// returned, in `requested`'s order -- so the first name given becomes the
/// default.
///
/// # Errors
///
/// Returns an error if a requested name doesn't match any discovered model,
/// or if the same name is requested more than once (which would otherwise
/// load and register the same model twice under one name).
fn resolve_requested_models<'a>(
    discovered: &'a [DiscoveredModel],
    requested: &[String],
    flag_name: &str,
) -> Result<Vec<&'a DiscoveredModel>> {
    if requested.is_empty() {
        return Ok(discovered.iter().collect());
    }

    let mut seen = std::collections::HashSet::with_capacity(requested.len());
    let mut resolved = Vec::with_capacity(requested.len());
    for name in requested {
        if !seen.insert(name.as_str()) {
            anyhow::bail!("model '{name}' specified more than once in {flag_name}");
        }
        let model = discovered.iter().find(|m| &m.name == name).ok_or_else(|| {
            let available: Vec<&str> = discovered.iter().map(|m| m.name.as_str()).collect();
            anyhow::anyhow!(
                "model '{name}' not found; available: {}",
                available.join(", ")
            )
        })?;
        resolved.push(model);
    }
    Ok(resolved)
}

/// Resolve the `--model-tts` names an operator requested against the models
/// [`discover_models`] found under `<model-path>/tts`. The first name given
/// becomes the default voice; see [`resolve_requested_models`] for details.
///
/// # Errors
///
/// See [`resolve_requested_models`].
pub fn resolve_models_to_load<'a>(
    discovered: &'a [DiscoveredModel],
    requested: &[String],
) -> Result<Vec<&'a DiscoveredModel>> {
    resolve_requested_models(discovered, requested, "--model-tts")
}

/// Resolve the `--model-asr` names an operator requested against the models
/// [`discover_asr_models`] found under `<model-path>/asr`. The first name
/// given becomes the default ASR model; see [`resolve_requested_models`] for
/// details.
///
/// # Errors
///
/// See [`resolve_requested_models`].
pub fn resolve_asr_models_to_load<'a>(
    discovered: &'a [DiscoveredModel],
    requested: &[String],
) -> Result<Vec<&'a DiscoveredModel>> {
    resolve_requested_models(discovered, requested, "--model-asr")
}

// ─────────────────────────────────────────────────────────────
//  Factory
// ─────────────────────────────────────────────────────────────

/// Resolve `ModelType::Auto` to a concrete TTS type.
///
/// If `model_type` isn't `Auto`, it's returned unchanged; [`create_tts`]
/// performs the actual TTS/ASR family check at construction time.
///
/// # Errors
///
/// Returns an error if `model_type` is `Auto` and [`detect_model_type`]
/// identifies the model at `model_path` as an ASR type. Auto-detection has
/// no way to know a TTS model was expected, so a wrong-family match must be
/// rejected here rather than deferred to [`create_tts`], where it would
/// otherwise report a confusing "already resolved" mismatch.
pub fn resolve_tts(model_type: ModelType, model_path: &str) -> Result<ModelType> {
    if model_type != ModelType::Auto {
        return Ok(model_type);
    }
    let detected = detect_model_type(model_path);
    if detected.is_asr() {
        anyhow::bail!(
            "auto-detected '{model_path}' as ASR model type '{}', but a TTS model was expected",
            detected.display_name()
        );
    }
    Ok(detected)
}

/// Resolve `ModelType::Auto` to a concrete ASR type.
///
/// If `model_type` isn't `Auto`, it's returned unchanged; [`create_asr`]
/// performs the actual TTS/ASR family check at construction time.
///
/// # Errors
///
/// Returns an error if `model_type` is `Auto` and [`detect_model_type`]
/// identifies the model at `model_path` as a TTS type. Auto-detection has
/// no way to know an ASR model was expected, so a wrong-family match must be
/// rejected here rather than deferred to [`create_asr`], where it would
/// otherwise report a confusing "already resolved" mismatch.
pub fn resolve_asr(model_type: ModelType, model_path: &str) -> Result<ModelType> {
    if model_type != ModelType::Auto {
        return Ok(model_type);
    }
    let detected = detect_model_type(model_path);
    if detected.is_tts() {
        anyhow::bail!(
            "auto-detected '{model_path}' as TTS model type '{}', but an ASR model was expected",
            detected.display_name()
        );
    }
    Ok(detected)
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
        ModelType::Qwen3ASR => anyhow::bail!(
            "ModelType::{} is an ASR model type; use create_asr() instead of create_tts()",
            model_type.display_name()
        ),
    }
}

/// Create an ASR model as a trait object.
///
/// Dispatches on [`ModelType`] and returns a `Box<dyn Asr + Send>`, allowing
/// callers (e.g. [`crate::engine::runtime::ModelRuntime`]) to work with any
/// ASR model through the [`crane::audio::Asr`] trait without depending on
/// concrete model types.
///
/// # Errors
///
/// Returns an error if `model_type` is `Auto` (must be resolved first) or a
/// TTS type, or if the model fails to load from `model_path`.
pub fn create_asr(
    model_type: ModelType,
    model_path: &str,
    device: &Device,
    dtype: &DType,
) -> Result<Box<dyn crane::audio::Asr + Send>> {
    tracing::info!("Creating ASR model: {:?}", model_type);

    match model_type {
        ModelType::Qwen3ASR => Ok(Box::new(crane_core::models::qwen3_asr::Model::new(
            model_path, device, dtype,
        )?)),
        ModelType::Auto => anyhow::bail!("ModelType::Auto must be resolved before create_asr()"),
        ModelType::Qwen3TTS | ModelType::VoxtralTTS => anyhow::bail!(
            "ModelType::{} is a TTS model type; use create_tts() instead of create_asr()",
            model_type.display_name()
        ),
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
    fn model_type_from_str_qwen3_asr_variants() {
        assert_eq!(ModelType::from_str("qwen3_asr"), ModelType::Qwen3ASR);
        assert_eq!(ModelType::from_str("qwen3asr"), ModelType::Qwen3ASR);
        assert_eq!(ModelType::from_str("qwen3-asr"), ModelType::Qwen3ASR);
        assert_eq!(ModelType::from_str("asr"), ModelType::Qwen3ASR);
        assert_eq!(ModelType::from_str("QWEN3_ASR"), ModelType::Qwen3ASR);
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
        assert_eq!(ModelType::Qwen3ASR.display_name(), "qwen3_asr");
    }

    #[test]
    fn model_type_is_tts_and_is_asr() {
        assert!(ModelType::Qwen3TTS.is_tts());
        assert!(ModelType::VoxtralTTS.is_tts());
        assert!(!ModelType::Qwen3ASR.is_tts());
        assert!(!ModelType::Auto.is_tts());

        assert!(ModelType::Qwen3ASR.is_asr());
        assert!(!ModelType::Qwen3TTS.is_asr());
        assert!(!ModelType::VoxtralTTS.is_asr());
        assert!(!ModelType::Auto.is_asr());
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
    fn detect_from_config_json_model_type_qwen3_asr() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"model_type": "qwen3_asr"}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen3ASR);
    }

    #[test]
    fn detect_from_config_json_architectures_qwen3_asr() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(
            &config,
            r#"{"architectures": ["Qwen3ASRForConditionalGeneration"]}"#,
        )
        .unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen3ASR);
    }

    #[test]
    fn detect_path_heuristic_qwen3_asr() {
        let result = detect_model_type("/models/Qwen3-ASR-0.6B-hf");
        assert_eq!(result, ModelType::Qwen3ASR);
    }

    #[test]
    fn detect_fallback_unknown_defaults_to_qwen3_tts() {
        let dir = tempfile::tempdir().unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen3TTS);
    }

    // ── resolve_tts / resolve_asr ──

    #[test]
    fn resolve_tts_auto_delegates_to_detect() {
        let result = resolve_tts(ModelType::Auto, "/models/Voxtral-4B-TTS-2603").unwrap();
        assert_eq!(result, ModelType::VoxtralTTS);
    }

    #[test]
    fn resolve_tts_explicit_type_is_passthrough() {
        let result = resolve_tts(ModelType::Qwen3TTS, "/models/whatever").unwrap();
        assert_eq!(result, ModelType::Qwen3TTS);
    }

    #[test]
    fn resolve_tts_rejects_auto_detected_asr() {
        let err = resolve_tts(ModelType::Auto, "/models/Qwen3-ASR-0.6B-hf").unwrap_err();
        assert!(err.to_string().contains("TTS model was expected"));
    }

    #[test]
    fn resolve_asr_auto_delegates_to_detect() {
        let result = resolve_asr(ModelType::Auto, "/models/Qwen3-ASR-0.6B-hf").unwrap();
        assert_eq!(result, ModelType::Qwen3ASR);
    }

    #[test]
    fn resolve_asr_explicit_type_is_passthrough() {
        let result = resolve_asr(ModelType::Qwen3ASR, "/models/whatever").unwrap();
        assert_eq!(result, ModelType::Qwen3ASR);
    }

    #[test]
    fn resolve_asr_rejects_auto_detected_tts() {
        let err = resolve_asr(ModelType::Auto, "/models/Voxtral-4B-TTS-2603").unwrap_err();
        assert!(err.to_string().contains("ASR model was expected"));
    }

    #[test]
    fn resolve_asr_rejects_unrecognized_path_defaulting_to_tts() {
        // detect_model_type falls back to Qwen3TTS for anything it can't
        // recognize; resolve_asr must reject that fallback rather than
        // silently handing an ASR caller a TTS type.
        let err = resolve_asr(ModelType::Auto, "/models/mystery-model").unwrap_err();
        assert!(err.to_string().contains("ASR model was expected"));
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

    #[test]
    fn create_tts_rejects_asr_type() {
        let result = create_tts(
            ModelType::Qwen3ASR,
            "/models/whatever",
            &Device::Cpu,
            &DType::F32,
        );
        let Err(err) = result else {
            panic!("expected create_tts to reject ModelType::Qwen3ASR");
        };
        assert!(err.to_string().contains("create_asr()"));
    }

    // ── create_asr ──

    #[test]
    fn create_asr_rejects_auto() {
        let result = create_asr(
            ModelType::Auto,
            "/models/whatever",
            &Device::Cpu,
            &DType::F32,
        );
        let Err(err) = result else {
            panic!("expected create_asr to reject ModelType::Auto");
        };
        assert!(err.to_string().contains("must be resolved"));
    }

    #[test]
    fn create_asr_rejects_tts_type() {
        let result = create_asr(
            ModelType::Qwen3TTS,
            "/models/whatever",
            &Device::Cpu,
            &DType::F32,
        );
        let Err(err) = result else {
            panic!("expected create_asr to reject ModelType::Qwen3TTS");
        };
        assert!(err.to_string().contains("create_tts()"));

        let result = create_asr(
            ModelType::VoxtralTTS,
            "/models/whatever",
            &Device::Cpu,
            &DType::F32,
        );
        let Err(err) = result else {
            panic!("expected create_asr to reject ModelType::VoxtralTTS");
        };
        assert!(err.to_string().contains("create_tts()"));
    }

    // ── probe_model_type ──

    #[test]
    fn probe_from_config_json_qwen3_tts() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "qwen3_tts"}"#,
        )
        .unwrap();
        assert_eq!(probe_model_type(dir.path()), Some(ModelType::Qwen3TTS));
    }

    #[test]
    fn probe_from_params_json_voxtral() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("params.json"),
            r#"{"model_type": "voxtral_tts"}"#,
        )
        .unwrap();
        assert_eq!(probe_model_type(dir.path()), Some(ModelType::VoxtralTTS));
    }

    #[test]
    fn probe_from_config_json_qwen3_asr() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "qwen3_asr"}"#,
        )
        .unwrap();
        assert_eq!(probe_model_type(dir.path()), Some(ModelType::Qwen3ASR));
    }

    #[test]
    fn probe_from_config_json_architectures_qwen3_asr() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"architectures": ["Qwen3ASRForConditionalGeneration"]}"#,
        )
        .unwrap();
        assert_eq!(probe_model_type(dir.path()), Some(ModelType::Qwen3ASR));
    }

    #[test]
    fn probe_empty_dir_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(probe_model_type(dir.path()), None);
    }

    #[test]
    fn probe_unrecognized_config_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "unknown"}"#,
        )
        .unwrap();
        assert_eq!(probe_model_type(dir.path()), None);
    }

    #[test]
    fn probe_path_heuristic_not_used() {
        // No config files exist here, so unlike `detect_model_type`, the
        // "voxtral" substring in the path must not cause a match.
        assert_eq!(
            probe_model_type(Path::new("/nonexistent/Voxtral-4B-TTS-2603")),
            None
        );
    }

    // ── discover_models ──

    #[test]
    fn discover_models_finds_both_types() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("qwen")).unwrap();
        std::fs::write(
            dir.path().join("qwen/config.json"),
            r#"{"model_type": "qwen3_tts"}"#,
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("voxtral")).unwrap();
        std::fs::write(
            dir.path().join("voxtral/params.json"),
            r#"{"model_type": "voxtral_tts"}"#,
        )
        .unwrap();

        let models = discover_models(dir.path()).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "qwen");
        assert_eq!(models[0].model_type, ModelType::Qwen3TTS);
        assert_eq!(models[1].name, "voxtral");
        assert_eq!(models[1].model_type, ModelType::VoxtralTTS);
    }

    #[test]
    fn discover_models_skips_asr_models() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("qwen-tts")).unwrap();
        std::fs::write(
            dir.path().join("qwen-tts/config.json"),
            r#"{"model_type": "qwen3_tts"}"#,
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("qwen-asr")).unwrap();
        std::fs::write(
            dir.path().join("qwen-asr/config.json"),
            r#"{"model_type": "qwen3_asr"}"#,
        )
        .unwrap();

        let models = discover_models(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "qwen-tts");
        assert_eq!(models[0].model_type, ModelType::Qwen3TTS);
    }

    #[test]
    fn discover_models_skips_non_model_dirs_and_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join("readme.txt"), "hello").unwrap();
        std::fs::create_dir(dir.path().join("model")).unwrap();
        std::fs::write(
            dir.path().join("model/config.json"),
            r#"{"model_type": "qwen3_tts"}"#,
        )
        .unwrap();

        let models = discover_models(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "model");
    }

    #[test]
    fn discover_models_sorted_alphabetically() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["c", "a", "b"] {
            let sub = dir.path().join(name);
            std::fs::create_dir(&sub).unwrap();
            std::fs::write(sub.join("config.json"), r#"{"model_type": "qwen3_tts"}"#).unwrap();
        }

        let models = discover_models(dir.path()).unwrap();
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn discover_models_empty_dir_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(discover_models(dir.path()).unwrap(), vec![]);
    }

    #[test]
    fn discover_models_nonexistent_dir_errors() {
        let result = discover_models(Path::new("/nonexistent/parent/dir"));
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn discover_models_follows_symlinked_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real-model");
        std::fs::create_dir(&real).unwrap();
        std::fs::write(real.join("config.json"), r#"{"model_type": "qwen3_tts"}"#).unwrap();

        let link = dir.path().join("linked-model");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let models = discover_models(dir.path()).unwrap();
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["linked-model", "real-model"]);
    }

    // ── discover_asr_models ──

    #[test]
    fn discover_asr_models_finds_asr_skips_tts() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("qwen-tts")).unwrap();
        std::fs::write(
            dir.path().join("qwen-tts/config.json"),
            r#"{"model_type": "qwen3_tts"}"#,
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("qwen-asr")).unwrap();
        std::fs::write(
            dir.path().join("qwen-asr/config.json"),
            r#"{"model_type": "qwen3_asr"}"#,
        )
        .unwrap();

        let models = discover_asr_models(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "qwen-asr");
        assert_eq!(models[0].model_type, ModelType::Qwen3ASR);
    }

    #[test]
    fn discover_asr_models_sorted_alphabetically() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["c", "a", "b"] {
            let sub = dir.path().join(name);
            std::fs::create_dir(&sub).unwrap();
            std::fs::write(sub.join("config.json"), r#"{"model_type": "qwen3_asr"}"#).unwrap();
        }

        let models = discover_asr_models(dir.path()).unwrap();
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn discover_asr_models_empty_when_only_tts_present() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("qwen-tts")).unwrap();
        std::fs::write(
            dir.path().join("qwen-tts/config.json"),
            r#"{"model_type": "qwen3_tts"}"#,
        )
        .unwrap();

        let models = discover_asr_models(dir.path()).unwrap();
        assert_eq!(models, vec![]);
    }

    #[test]
    fn discover_asr_models_empty_dir_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(discover_asr_models(dir.path()).unwrap(), vec![]);
    }

    #[test]
    fn discover_asr_models_nonexistent_dir_errors() {
        let result = discover_asr_models(Path::new("/nonexistent/parent/dir"));
        assert!(result.is_err());
    }

    // ── resolve_models_to_load ──

    fn discovered_fixture() -> Vec<DiscoveredModel> {
        vec![
            DiscoveredModel {
                path: PathBuf::from("/models/alpha"),
                name: "alpha".to_string(),
                model_type: ModelType::Qwen3TTS,
            },
            DiscoveredModel {
                path: PathBuf::from("/models/beta"),
                name: "beta".to_string(),
                model_type: ModelType::VoxtralTTS,
            },
            DiscoveredModel {
                path: PathBuf::from("/models/gamma"),
                name: "gamma".to_string(),
                model_type: ModelType::Qwen3TTS,
            },
        ]
    }

    #[test]
    fn resolve_models_to_load_empty_requested_returns_all() {
        let discovered = discovered_fixture();
        let resolved = resolve_models_to_load(&discovered, &[]).unwrap();
        let names: Vec<&str> = resolved.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn resolve_models_to_load_single_match() {
        let discovered = discovered_fixture();
        let requested = vec!["beta".to_string()];
        let resolved = resolve_models_to_load(&discovered, &requested).unwrap();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].name, "beta");
    }

    #[test]
    fn resolve_models_to_load_custom_order() {
        let discovered = discovered_fixture();
        let requested = vec!["gamma".to_string(), "alpha".to_string()];
        let resolved = resolve_models_to_load(&discovered, &requested).unwrap();
        let names: Vec<&str> = resolved.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["gamma", "alpha"]);
    }

    #[test]
    fn resolve_models_to_load_unknown_name_errors() {
        let discovered = discovered_fixture();
        let requested = vec!["nope".to_string()];
        let err = resolve_models_to_load(&discovered, &requested).unwrap_err();
        assert!(err.to_string().contains("nope"));
        assert!(err.to_string().contains("alpha, beta, gamma"));
    }

    #[test]
    fn resolve_models_to_load_duplicate_name_errors() {
        let discovered = discovered_fixture();
        let requested = vec!["alpha".to_string(), "alpha".to_string()];
        let err = resolve_models_to_load(&discovered, &requested).unwrap_err();
        assert!(err.to_string().contains("more than once"));
    }

    // ── resolve_asr_models_to_load ──
    //
    // `resolve_models_to_load`'s tests above exercise the shared
    // `resolve_requested_models` logic; these just confirm the ASR wrapper
    // passes through its own `--model-asr` flag name in error messages.

    fn discovered_asr_fixture() -> Vec<DiscoveredModel> {
        vec![DiscoveredModel {
            path: PathBuf::from("/models/alpha-asr"),
            name: "alpha-asr".to_string(),
            model_type: ModelType::Qwen3ASR,
        }]
    }

    #[test]
    fn resolve_asr_models_to_load_empty_requested_returns_all() {
        let discovered = discovered_asr_fixture();
        let resolved = resolve_asr_models_to_load(&discovered, &[]).unwrap();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].name, "alpha-asr");
    }

    #[test]
    fn resolve_asr_models_to_load_duplicate_name_errors_mentions_model_asr_flag() {
        let discovered = discovered_asr_fixture();
        let requested = vec!["alpha-asr".to_string(), "alpha-asr".to_string()];
        let err = resolve_asr_models_to_load(&discovered, &requested).unwrap_err();
        assert!(err.to_string().contains("more than once in --model-asr"));
    }
}
