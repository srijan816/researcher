-- Websurfx config for the app2 deep-research stack.
-- This service is private on the Docker network and feeds the AI-Q web adapter.

logging = true
debug = false
threads = 8

port = "8080"
binding_ip = "0.0.0.0"
-- Production mode adds random delay for human HTML searches. Our Docker patch
-- skips that delay for json=true agent calls so AIQ research stays fast.
production_use = true
-- 6s: most engines answer in 1-4s; waiting the full 10s for stragglers made
-- every uncached discovery wave cost 10s flat.
request_timeout = 6
tcp_connection_keep_alive = 30
pool_idle_connection_timeout = 30
rate_limiter = {
	number_of_requests = 120,
	time_limit = 1,
}
https_adaptive_window_size = true
operating_system_tls_certificates = true
number_of_https_connections = 24
client_connection_keep_alive = 120

safe_search = 2

colorscheme = "catppuccin-mocha"
theme = "simple"
animation = nil

redis_url = "redis://websurfx-redis:6379"
-- 2h: concurrent/overlapping research jobs re-issue near-identical queries;
-- longer reuse also relieves upstream engine rate limits (429s).
cache_expiry_time = 7200
http_cache_expiry_time = 120

-- Searx uses WEBSURFX_SEARX_URL from the container env. App2 points it to the
-- private SearXNG service at http://searxng:8080 via a small Websurfx patch.
upstream_search_engines = {
	DuckDuckGo = true,
	Searx = true,
	Brave = true,
	Startpage = true,
	LibreX = true,
	Mojeek = true,
	Bing = true,
	Qwant = true,
	Wikipedia = false,
	Yahoo = true,
	SepiaSearch = true,
}

local proxy_env = os.getenv("WEBSURFX_PROXY")
if proxy_env == "" then
	proxy_env = nil
end
proxy = proxy_env
