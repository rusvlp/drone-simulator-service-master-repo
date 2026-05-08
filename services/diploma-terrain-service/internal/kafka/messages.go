package kafka

type GenerationRequest struct {
	JobID       string  `json:"job_id"`
	UserID      string  `json:"user_id"`
	ImageBase64 string  `json:"image_base64"`
	ScaleZ      float32 `json:"scale_z"`
	YUp         bool    `json:"y_up"`
	TextureMode string  `json:"texture_mode"`
}

type GenerationResult struct {
	JobID      string `json:"job_id"`
	Status     string `json:"status"`
	ModelURL   string `json:"model_url"`
	TextureURL string `json:"texture_url"`
	TreesURL   string `json:"trees_url"`
	Error      string `json:"error"`
}
