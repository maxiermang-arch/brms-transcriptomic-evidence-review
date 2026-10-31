# 阶段 3 · 步骤 17：ABIS 改用真 TPM 输入（官方要求），另存对照版本
# 议题 A · 2026-09-13
# 背景：Monaco 2019 (Cell Reports 26:1627) 的 sigmatrixRNAseq 在 TPM 尺度上校准。
#       步骤 13 用 CPM → 列和 9.5–48.3；本脚本用步骤 15(v4) 的真 TPM → 列和 2.9–23.1。
#       两次都不接近 100，原因见文档 8.2 与「局限」：① 签名基因只覆盖 55–65%；
#       ② 长度用基因跨度（含内含子）→ 系统性低估 TPM；③ bulk 脑转移组织里免疫细胞
#       本就是少数（10–40%），"和=100" 并不是本场景的期望值。
#       → **结论：ABIS 只能作相对量使用；其价值在于与我们自建免疫面板的秩相关。**

suppressMessages({library(MASS)})
script_arg <- grep("^--file=", commandArgs(), value=TRUE); ROOT <- dirname(dirname(normalizePath(sub("^--file=", "", script_arg[1])))); PU <- file.path(ROOT,"data/purity")
setwd(PU)

run_abis <- function(linear_file){
  sig <- read.delim("sigmatrixRNAseq.txt", check.names=FALSE, row.names=1)
  d   <- read.delim(linear_file, check.names=FALSE, row.names=1)
  genes <- intersect(rownames(d), rownames(sig))
  cat("    ABIS 可用基因:", length(genes), "/", nrow(sig), "\n")
  Dec <- apply(d[genes,,drop=FALSE], 2,
               function(x) coef(rlm(as.matrix(sig[genes,]), x, maxit=100))) * 100
  signif(Dec, 3)
}

for (co in c("A","B")){
  cat("\n===== 队列", co, "=====\n")
  d_tpm <- run_abis(sprintf("cohort%s_TPM.tsv", co))
  write.table(d_tpm, sprintf("cohort%s_ABIS_TPM.txt", co), sep="\t", quote=FALSE)
  s <- colSums(d_tpm)
  cat("    TPM 版列和: min=", round(min(s),1), " 中位=", round(median(s),1),
      " max=", round(max(s),1), "\n")
  d_cpm <- read.delim(sprintf("cohort%s_ABIS.txt", co), check.names=FALSE, row.names=1)
  s2 <- colSums(d_cpm)
  cat("    CPM 版列和: min=", round(min(s2),1), " 中位=", round(median(s2),1),
      " max=", round(max(s2),1), "\n")
}
cat("\n✅ ABIS(TPM) 完成\n")
