// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Andreas Schneider <asn@cryptomilk.org>

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    crane_wyoming::cli_main().await
}
