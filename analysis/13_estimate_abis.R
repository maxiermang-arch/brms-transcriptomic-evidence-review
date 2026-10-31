# 阶段 3 · 步骤 13：ESTIMATE（肿瘤纯度）+ ABIS（绝对免疫去卷积）
# 议题 A · 2026-09-13
# ESTIMATE = Yoshihara 2013, Nat Commun 4:2612（R 包官网实现，R-Forge 源码本地安装）
# ABIS     = Monaco 2019, Cell Reports 26:1627（严格照抄 server.R 的 rlm 实现）

suppressMessages({library(estimate); library(MASS)})
script_arg <- grep("^--file=", commandArgs(), value=TRUE); ROOT <- dirname(dirname(normalizePath(sub("^--file=", "", script_arg[1])))); PU <- file.path(ROOT,"data/purity")
setwd(PU)

## ---------- 工具：写 filterCommonGenes 期望的输入（普通 TSV，不是 GCT！） ----------
## 源码：read.table(input.f, header=TRUE, row.names=1, sep="\t", quote="")  —— 无 skip
write_tsv4estimate <- function(mat, file){
  df <- data.frame(Name=rownames(mat), Description="na", mat, check.names=FALSE)
  write.table(df, file, sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)
}

## ---------- ABIS（严格照 server.R 的 rlm 实现） ----------
run_abis <- function(linear_file){
  sig <- read.delim("sigmatrixRNAseq.txt", check.names=FALSE, row.names=1)
  d   <- read.delim(linear_file, check.names=FALSE, row.names=1)
  genes <- intersect(rownames(d), rownames(sig))
  cat("  ABIS 可用基因:", length(genes), "/ 签名矩阵", nrow(sig), "\n")
  Dec <- apply(d[genes,,drop=FALSE], 2,
               function(x) coef(rlm(as.matrix(sig[genes,]), x, maxit=100))) * 100
  Dec <- signif(Dec, 3)
  cat("  行名:", paste(rownames(Dec), collapse=" | "), "\n")
  Dec
}

## ---------- 主流程 ----------
for (co in c("A","B")){
  cat("\n===== 队列", co, "=====\n")
  logfile <- if (co=="A") "cohortA_log2CPM.tsv" else "cohortB_log2TMM.tsv"
  linfil  <- if (co=="A") "cohortA_linearCPM.tsv" else "cohortB_linearTMM.tsv"

  ## ---- ESTIMATE ----
  lg <- read.delim(logfile, check.names=FALSE, row.names=1)
  ## ⚠️ estimateScore 内部 read.delim 会把列名里的 - 换成 .（且首位非字母加 X）
  ##    → 改用安全列名 S001..，另存映射表
  orig <- colnames(lg)
  colnames(lg) <- sprintf("S%03d", seq_along(orig))
  write.table(data.frame(safe=colnames(lg), original=orig),
              sprintf("cohort%s_samplemap.txt", co),
              sep="\t", quote=FALSE, row.names=FALSE)
  gj <- sprintf("cohort%s_log2.tsv", co)
  write_tsv4estimate(as.matrix(lg), gj)          # filterCommonGenes 要普通 TSV
  fc <- sprintf("cohort%s_common.gct", co)       # 它自己输出真 GCT
  filterCommonGenes(input.f=gj, output.f=fc, id="GeneSymbol")
  es <- sprintf("cohort%s_estimate.gct", co)
  estimateScore(input.ds=fc, output.ds=es, platform="illumina")
  sc <- read.delim(es, skip=2, check.names=FALSE)
  write.table(sc, sprintf("cohort%s_ESTIMATE_raw.txt", co), sep="\t", quote=FALSE, row.names=FALSE)
  cat("  ESTIMATE 完成:", paste(dim(sc), collapse=" x "), "\n"); print(head(sc,3))

  ## ---- ABIS ----
  Dec <- run_abis(linfil)
  write.table(Dec, sprintf("cohort%s_ABIS.txt", co), sep="\t", quote=FALSE)
  cat("  ABIS 完成:", paste(dim(Dec), collapse=" x "), "\n")
  print(Dec[1:min(19,nrow(Dec)), 1:min(3,ncol(Dec))])
}

cat("\n✅ ESTIMATE + ABIS 全部完成\n")
