// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Andreas Schneider <asn@cryptomilk.org>

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    cw_say::cli_main().await
}
