// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Andreas Schneider <asn@cryptomilk.org>
//
// Based on the `engine` module of Crane's `crane-serve` crate
// (https://github.com/lucasjinreal/Crane), Copyright (c) 2024 Nicholas Jela,
// licensed under the MIT License.

//! Protocol-independent TTS/ASR model runtime, owned by crane-wyoming.
//!
//! Unlike the Crane monorepo's `crane::engine` (which also hosts the LLM
//! continuous-batching engine and VLM channels), this module only ever
//! deals with TTS and ASR models: crane-wyoming has no use for the rest.
//!
//! | Module          | Responsibility                                       |
//! |-----------------|-------------------------------------------------------|
//! | `model_factory` | TTS/ASR model type auto-detection and construction     |
//! | `runtime`       | `ModelRuntime` -- owns loaded TTS and ASR models      |
//! | `cache`         | `TtsCache` -- optional disk cache for TTS responses    |

pub mod cache;
pub mod model_factory;
pub mod runtime;

pub use cache::TtsCache;
pub use runtime::{AsrHandle, AsrTranscribeRequest, ModelRuntime, TtsGenerateRequest, TtsHandle};
