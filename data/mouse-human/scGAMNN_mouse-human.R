# Ref:  https://github.com/QSong-github/scGCN
GenerateMNN <- function(Dat1,Dat2,K){
  object1 <- CreateSeuratObject(counts=Dat1,project = "1",assay = "RNA",
                                min.cells = 0,min.features = 0,
                                names.field = 1,names.delim = "_")
  
  object2 <- CreateSeuratObject(counts=Dat2,project = "2",assay = "RNA",
                                min.cells = 0,min.features =0,names.field = 1,
                                names.delim = "_")
  
  objects <- list(object1,object2)    
  objects1 <- lapply(objects,function(obj){
    obj <- NormalizeData(obj,verbose=F)
    obj <- FindVariableFeatures(obj,
                                selection.method = "vst",
                                nfeatures = 2000,verbose=F)
    obj <- ScaleData(obj,features=rownames(obj),verbose=FALSE)
    obj <- RunPCA(obj, features=rownames(obj), verbose = FALSE)
    obj <- SCTransform(obj)
    return(obj)})
  #'  Inter-data graph  
  object.nn <- FindIntegrationAnchors(object.list = objects1,k.anchor=K,verbose=F)
  arc=object.nn@anchors
  d1.arc1=cbind(arc[arc[,4]==1,1],arc[arc[,4]==1,2],arc[arc[,4]==1,3]) 
  grp1=d1.arc1[d1.arc1[,3]>0,1:2]-1
  grp=cbind(grp1[,1],grp1[,2]+ncol(objects1[[1]]))
  
  return (grp)
}
#################
count.list <- readRDS("E:/scGAMNN/data/mouse-human/count.list.RDS")
label.list <- readRDS("E:/scGAMNN/data/mouse-human/label.list.RDS")

##-------
count_list=count.list
label_list=label.list
cells1=rownames(label_list[[1]])
omit1=cells1[which(label_list[[1]]=='endothelial')]
label_list[[1]]=label_list[[1]][-which(rownames(label_list[[1]])%in%omit1),]
count_list[[1]]=count_list[[1]][,-which(colnames(count_list[[1]])%in%omit1)]
label_list[[1]]=as.data.frame(label_list[[1]])
rownames(label_list[[1]])=cells1[-which(cells1 %in%omit1)]
cells2=rownames(label_list[[2]])
c1=cells2[which(label_list[[2]]=='macrophage')]
c2=cells2[which(label_list[[2]]=='delta')]
omit2=c(c1,c2)
label_list[[2]]=label_list[[2]][-which(rownames(label_list[[2]])%in%omit2),]
count_list[[2]]=count_list[[2]][,-which(colnames(count_list[[2]])%in%omit2)]
label_list[[2]]=as.data.frame(label_list[[2]])
rownames(label_list[[2]])=cells2[-which(cells2 %in%omit2)]
count.list=count_list
label.list=label_list

##---------
library(Seurat)
match<-GenerateMNN(count.list[[1]],count.list[[2]],5)
setwd("E:/scGAMNN/data/mouse-human/") 
write.csv(match,file='match.csv',quote=F,row.names=F)
write.csv(t(count.list[[1]]),file='Data1.csv',quote=F,row.names=T)
write.csv(t(count.list[[2]]),file='Data2.csv',quote=F,row.names=T)
write.csv(label.list[[1]],file='label1.csv',quote=F,row.names=F)
write.csv(label.list[[2]],file='label2.csv',quote=F,row.names=F)



