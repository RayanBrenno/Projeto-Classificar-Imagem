//! Axum gateway: POST /classify (multipart `image`) → resize 32×32 PNG → sidecar Python → JSON.

mod error;
mod state;

use std::net::SocketAddr;
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use image::ImageFormat;
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, instrument, warn};

use error::ApiError;
use state::AppState;

// DefaultBodyLimit rejeita o body inteiro antes do handler; MAX_UPLOAD_BYTES protege leituras por campo.
const MAX_UPLOAD_BYTES: usize = 5 * 1024 * 1024; // 5 MB

#[derive(Debug, Deserialize, Serialize)]
struct PredictResponse {
    class: String,
    confidence: f32,
    top_k: Vec<(String, f32)>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Usa info quando RUST_LOG não está definida.
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "classify_api=info,tower_http=info".into()))
        .init();

    let sidecar_url = std::env::var("SIDECAR_URL")
        .unwrap_or_else(|_| "http://localhost:8001".into());

    let bind: SocketAddr = std::env::var("BIND")
        .unwrap_or_else(|_| "0.0.0.0:8000".into())
        .parse()?;

    // Um único cliente compartilhado; reqwest gerencia pool de conexões internamente.
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;

    let state = AppState { sidecar_url, http };

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/classify", post(classify))
        .with_state(state)
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        // Qualquer origem/método/header — adequado para dev local, restringir em produção.
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(TraceLayer::new_for_http());

    info!(%bind, "iniciando classify-api");

    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// skip_all evita logar bytes brutos da imagem no span.
#[instrument(skip_all)]
async fn classify(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, ApiError> {
    let mut image_bytes: Option<bytes::Bytes> = None;
    let mut filename: Option<String> = None;

    while let Some(field) = multipart.next_field().await? {
        if field.name() == Some("image") {
            filename = field.file_name().map(|s| s.to_string());
            let data = field.bytes().await?;
            if data.len() > MAX_UPLOAD_BYTES {
                return Err(ApiError::PayloadTooLarge);
            }
            image_bytes = Some(data);
        }
    }

    let bytes = image_bytes.ok_or(ApiError::MissingImageField)?;
    info!(filename = ?filename, bytes = bytes.len(), "imagem recebida");

    // preprocess é CPU-bound (decode + resize Lanczos3); spawn_blocking evita bloquear o runtime async.
    let png = tokio::task::spawn_blocking(move || preprocess(&bytes))
        .await
        .map_err(|e| ApiError::Internal(anyhow::anyhow!(e)))?? // primeiro ? = JoinError, segundo ? = ApiError do preprocess

    ;

    let part = reqwest::multipart::Part::bytes(png)
        .file_name("input.png")
        .mime_str("image/png")
        .map_err(|e| ApiError::Internal(e.into()))?;
    let form = reqwest::multipart::Form::new().part("image", part);

    let resp = state
        .http
        .post(format!("{}/predict", state.sidecar_url))
        .multipart(form)
        .send()
        .await?;

    if !resp.status().is_success() {
        warn!(status = %resp.status(), "sidecar respondeu com erro");
        return Err(ApiError::Sidecar(resp.status().as_u16()));
    }

    let body: PredictResponse = resp.json().await?;
    Ok(Json(body))
}

// Decodifica qualquer formato suportado pela crate `image`, redimensiona para 32×32 (entrada CIFAR-10)
// e retorna os bytes como PNG. Projetada para rodar via spawn_blocking.
fn preprocess(bytes: &[u8]) -> Result<Vec<u8>, ApiError> {
    let img = image::load_from_memory(bytes).map_err(|_| ApiError::InvalidImage)?;
    let resized = img.resize_exact(32, 32, image::imageops::FilterType::Lanczos3);

    // Cursor<Vec<u8>> age como arquivo em memória; 8 KB cobre a maioria dos PNGs 32×32.
    let mut buf = std::io::Cursor::new(Vec::with_capacity(8 * 1024));
    resized
        .write_to(&mut buf, ImageFormat::Png)
        .map_err(|e| ApiError::Internal(e.into()))?;

    Ok(buf.into_inner())
}
