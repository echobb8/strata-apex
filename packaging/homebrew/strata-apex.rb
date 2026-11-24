class StrataApex < Formula
  desc "Native Apple Silicon Metal 4 LLM & continuous-batching swarm inference engine"
  homepage "https://github.com/echobb8/strata-apex"
  # Live formula with release SHA-256 is maintained in https://github.com/echobb8/homebrew-tap
  url "https://github.com/echobb8/strata-apex/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "SET_IN_HOMEBREW_TAP_ON_RELEASE"
  license "MIT"
  head "https://github.com/echobb8/strata-apex.git", branch: "main"

  # Minimum macOS 14 (Sonoma) for Metal 3.1; macOS 26 (Tahoe)+ recommended for Metal 4 NAX tensor ops
  depends_on macos: :sonoma
  depends_on arch: :arm64

  def install
    system "make", "cli"
    bin.install "build/strata-apex"
    bin.install "scripts/bench_showcase.sh" => "strata-apex-bench"
  end

  def caveats
    <<~EOS
      Check Apple Silicon readiness and UMA budget:
        strata-apex doctor

      Download & pack Inkling-2-Mini (~14.2 GiB + 0.6 GiB Vision):
        strata-apex pull inkling-2-mini

      Start an OpenAI-compatible server (solo or local coding swarm):
        strata-apex serve inkling-2-mini --max-batch 8 --max-seq-len 16384

      Configure OpenCode automatically:
        strata-apex setup-opencode

      Run the STRATA // APEX benchmark card (including concurrent swarm stats):
        strata-apex-bench inkling --swarm-only
    EOS
  end

  test do
    assert_match "strata-apex 0.1.0", shell_output("#{bin}/strata-apex --version")
    assert_match "Strata Apex", shell_output("#{bin}/strata-apex doctor")
  end
end
