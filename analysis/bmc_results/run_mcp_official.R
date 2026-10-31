args<-commandArgs(TRUE)
source(args[1])
genes<-read.delim(args[2],check.names=FALSE,stringsAsFactors=FALSE)
for(co in c('A','B')) {
 x<-as.matrix(read.csv(file.path(args[3],paste0('mcp_expression_',co,'.csv')),row.names=1,check.names=FALSE))
 s<-MCPcounter.estimate(x,featuresType='HUGO_symbols',genes=genes)
 write.csv(t(s),file.path(args[3],paste0('mcp_official_',co,'.csv')))
}
capture.output(sessionInfo(),file=file.path(args[3],'MCP_R_sessionInfo.txt'))
