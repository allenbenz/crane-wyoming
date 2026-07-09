//! Protocol-independent TTS model runtime, owned by crane-wyoming.
//!
//! Unlike the Crane monorepo's `crane::engine` (which also hosts the LLM
//! continuous-batching engine and VLM channels), this module only ever
//! deals with TTS models: crane-wyoming has no use for the rest.
//!
//! | Module          | Responsibility                                   |
//! |-----------------|--------------------------------------------------|
//! | `model_factory` | TTS model type auto-detection and construction    |
//! | `runtime`       | `ModelRuntime` -- owns loaded TTS models          |
//! | `cache`         | `TtsCache` -- optional disk cache for TTS responses |

pub mod cache;
pub mod model_factory;
pub mod runtime;

pub use cache::TtsCache;
pub use runtime::{ModelRuntime, TtsGenerateRequest, TtsHandle};
