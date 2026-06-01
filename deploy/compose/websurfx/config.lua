-- Websurfx config for the app2 deep-research stack.
-- This service is private on the Docker network and feeds the AI-Q web adapter.

logging = true
debug = false
threads = 8

port = "8080"
binding_ip = "0.0.0.0"
production_use = false
request_timeout = 8
tcp_connection_keep_alive = 30
pool_idle_connection_timeout = 30
rate_limiter = {
	number_of_requests = 40,
	time_limit = 3,
}
https_adaptive_window_size = true
operating_system_tls_certificates = true
number_of_https_connections = 12
client_connection_keep_alive = 120

safe_search = 0

colorscheme = "catppuccin-mocha"
theme = "simple"
animation = nil

redis_url = "redis://127.0.0.1:8082"
cache_expiry_time = 900
http_cache_expiry_time = 120

upstream_search_engines = {
	DuckDuckGo = true,
	Searx = false,
	Brave = true,
	Startpage = false,
	LibreX = false,
	Mojeek = false,
	Bing = true,
	Qwant = false,
	Wikipedia = true,
	Yahoo = false,
	SepiaSearch = false,
}

proxy = nil
