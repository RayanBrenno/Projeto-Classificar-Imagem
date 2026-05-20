use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use thiserror::Error;

// thiserror gera impl Display e impl Error a partir dos atributos #[error] de cada variante.
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("campo `image` ausente no multipart")]
    MissingImageField,

    #[error("imagem inválida ou formato não suportado")]
    InvalidImage,

    // Dupla checagem: DefaultBodyLimit cobre o body inteiro; PayloadTooLarge cobre leitura por campo.
    #[error("upload excede o tamanho máximo (5MB)")]
    PayloadTooLarge,

    // #[from] gera From<MultipartError> → ApiError, permitindo usar `?` diretamente nos handlers.
    #[error("multipart parse error: {0}")]
    Multipart(#[from] axum::extract::multipart::MultipartError),

    #[error("falha ao falar com o sidecar: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("sidecar respondeu HTTP {0}")]
    Sidecar(u16),

    #[error("erro interno: {0}")]
    Internal(#[from] anyhow::Error),
}

// Retorna uma resposta HTTP para cada erro
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code) = match &self {
            ApiError::MissingImageField => (StatusCode::BAD_REQUEST,          "missing_image"),
            ApiError::InvalidImage      => (StatusCode::UNPROCESSABLE_ENTITY, "invalid_image"),
            ApiError::PayloadTooLarge   => (StatusCode::PAYLOAD_TOO_LARGE,    "payload_too_large"),
            ApiError::Multipart(_)      => (StatusCode::BAD_REQUEST,          "multipart_error"),
            ApiError::Reqwest(_)        => (StatusCode::BAD_GATEWAY,          "sidecar_unreachable"),
            ApiError::Sidecar(_)        => (StatusCode::BAD_GATEWAY,          "sidecar_error"),
            ApiError::Internal(_)       => (StatusCode::INTERNAL_SERVER_ERROR, "internal"),
        };

        tracing::error!(error = %self, "request failed");

        let body = Json(json!({
            "error": code,
            "message": self.to_string(),
        }));

        (status, body).into_response()
    }
}
