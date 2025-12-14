class MlxServe < Formula
  include Language::Python::Virtualenv

  desc "MLX-based embedding and reranking server with OpenAI-compatible API"
  homepage "https://github.com/menaje/mlx-serve"
  url "https://github.com/menaje/mlx-serve/archive/refs/tags/v0.2.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"
  head "https://github.com/menaje/mlx-serve.git", branch: "main"

  depends_on "python@3.11"
  depends_on :macos

  def install
    virtualenv_install_with_resources
  end

  def post_install
    # Create default config directory
    (var/"mlx-serve").mkpath
  end

  service do
    run [opt_bin/"mlx-serve", "start", "--foreground"]
    keep_alive true
    working_dir var/"mlx-serve"
    log_path var/"log/mlx-serve.log"
    error_log_path var/"log/mlx-serve.error.log"
    environment_variables PATH: std_service_path_env
  end

  def caveats
    <<~EOS
      To start mlx-serve as a background service:
        brew services start mlx-serve

      Or run in foreground:
        mlx-serve start --foreground

      Configuration file location:
        ~/.mlx-serve/config.yaml

      Generate example config:
        mlx-serve config --example > ~/.mlx-serve/config.yaml

      Download a model:
        mlx-serve pull Qwen/Qwen3-Embedding-0.6B

      API documentation available at:
        http://localhost:8000/docs
    EOS
  end

  test do
    assert_match "mlx-serve version", shell_output("#{bin}/mlx-serve --version")
  end
end
