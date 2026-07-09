#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    cw_say::cli_main().await
}
