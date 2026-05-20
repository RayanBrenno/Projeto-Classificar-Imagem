// Estado global clonado para cada handler pelo Axum — todos os campos devem ser Clone e baratos de clonar.
#[derive(Clone)]
pub struct AppState {
    pub sidecar_url: String,   // padrão: http://localhost:8001 (SIDECAR_URL)
    pub http: reqwest::Client, // pool de conexões via Arc interno; caro criar por request
}
