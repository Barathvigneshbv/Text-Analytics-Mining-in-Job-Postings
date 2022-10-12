

library(tm)
library(tmap)
library(SnowballC)
library(ggplot2)
library(DT)
library(plyr)
library(tibble)
library(RColorBrewer)
library(wordcloud)
library(stringr)
library(textstem)
library(RWeka)
library(qdap)
library(RColorBrewer)
library(udpipe)
library(lattice)
library(filematrix)


## Data Loading



# Reading the dataset

data.text  <- read.csv("C:/Users/india/Desktop/SEM 6/Data Warehousing and Mining/Text-Analytics-Mining-master/DataScientistjobs.csv")  




# Creating a Corpus

## Corpus Preparation


# Create corpus
corpus = Corpus(VectorSource(data.text$Description))
# Look at corpus
print(corpus)

cleanCorpus <- function(corpus){
  # Converting to lower-case
  corpus.cleaned <- tm_map(corpus,tolower)
  # Removing punctuation
  corpus.cleaned <- tm_map(corpus.cleaned, removePunctuation)
  # Looking at stop words 
  v_stopwords <- c(stopwords("en"),"will","etc","build","using","usemploymentcompliancecgicom","unless","vary","reason","routed","recruited")
  corpus.cleaned <- tm_map(corpus.cleaned, removeWords,v_stopwords )
  #Eliminating extra whitespace
  corpus.cleaned <- tm_map(corpus.cleaned, stripWhitespace)
  #Removing numbers
  corpus.cleaned <- tm_map(corpus.cleaned, removeNumbers)
  #Stemming/Lemmatizing document 
  corpus.cleaned <- tm_map(corpus.cleaned, lemmatize_strings)
  #Removing special characters 
  corpus.cleaned <- tm_map(corpus.cleaned, str_replace_all,"[^[:alnum:]]", " ")
  #corpus[[1]]$content
  return(corpus.cleaned)
}
corpus <- cleanCorpus(corpus)



#convert to document term matrix
corpustdm <- TermDocumentMatrix(corpus)
#corpustdm <- TermDocumentMatrix(corpus)corp
#print(corpustdm) 
#print(dimnames(corpustdm)$Terms)
#dim(corpustdm)
inspect(corpustdm[1:10,1:10])



# Text Analysis

##Frequency words

#collapse matrix by summing over columns
set.seed(43)
colS <- colSums(as.matrix(t(corpustdm)))

#total number of terms
length(colS)
# create sort order (asc)
ord <- order(colS,decreasing=TRUE)
colS[head(ord)]
colS[tail(ord)]
findFreqTerms(corpustdm, lowfreq=50)

findAssocs(corpustdm, "analysis",0.7)
findAssocs(corpustdm, "degree",0.7)


# Text Visualization




m <- as.matrix(corpustdm)
corpusdf <- data.frame(m)

v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)

corpusdf <- rownames_to_column(corpusdf)
corpusdf <- rename.(corpusdf, c("rowname"="word"))
corpusdf$wordcount <- rowSums(corpusdf[-1])
corpusdf <- corpusdf[order(-corpusdf$wordcount),]



#plotting histogram
ggplot(subset(corpusdf, wordcount>43), aes(reorder(word, -wordcount), wordcount)) +
   geom_bar(stat = "identity", fill = "darkgreen", alpha = 0.7) +
  xlab("Words with the highest frequency") +
  ylab("Frequency") +
  ggtitle("Understanding word frequencies") +
  theme_bw() +
  theme(axis.text.x=element_text(angle=60, hjust=0.9), plot.title = element_text(hjust=0.5)) +
 scale_fill_brewer() 



pal <- colorRampPalette(colors = c("darkgreen", "lightgreen"))(10)

barplot(d[1:10,]$freq, las = 2, names.arg = d[1:10,]$word,
         main ="Top 10 Most Frequent Words",
        ylab = "Frequency", col = pal,  border = NA)




# wordcloud(names(colS), colS, colors = brewer.pal(6, 'Dark2'),random.order=FALSE, rot.per= 0.35, max.words = 100)



#since all the rows are same and numeric,we can add them up to get the total value
#sort it based on the number

pal <- brewer.pal(9,"RdYlGn")
pal <- pal[-(1:2)]

set.seed(142)
wordcloud(word=corpusdf$word, freq= corpusdf$wordcount,  colors = brewer.pal(6, "Dark2"), random.order=FALSE, rot.per= 0.35, max.words = 150)




## Universal Word Classes {.tabset}




ud_model <- udpipe_download_model(language = "english")
ud_model <- udpipe_load_model(ud_model$file_model)
ud_model <- udpipe_load_model(file = "C:/Users/india/Documents/english-ewt-ud-2.5-191206.udpipe")


x <- udpipe_annotate(ud_model, x = data.text$Description, doc_id = data.text$Company)
x <- as.data.frame(x)


stats <- txt_freq(x$upos)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = stats, col = "lightgreen", 
         main = "UPOS (Universal Parts of Speech)\n frequency of occurrence", 
         xlab = "Freq")


## ADJECTIVES
stats <- subset(x, upos %in% c("ADJ")) 
stats <- txt_freq(stats$token)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 20), col = "purple", 
         main = "Most occurring adjectives", xlab = "Freq")

##NOUNS
stats <- subset(x, upos %in% c("NOUN")) 
stats <- txt_freq(stats$token)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 20), col = "blue", 
                     main = "Most occurring nouns", xlab = "Freq")




## VERB
stats <- subset(x, upos %in% c("VERB")) 
stats <- txt_freq(stats$token)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 20), col = "gold", 
         main = "Most occurring Verbs", xlab = "Freq")


## Using RAKE
#Rapid Automatic Keyword Extraction (RAKE) is an algorithm to automatically extract keywords from documents.
#More info on https://www.thinkinfi.com/2018/09/keyword-extraction-using-rake-in-python.html

stats <- keywords_rake(x = x, term = "lemma", group = "doc_id", 
                       relevant = x$upos %in% c("NOUN", "ADJ"))
stats$key <- factor(stats$keyword, levels = rev(stats$keyword))
barchart(key ~ rake, data = head(subset(stats, freq > 3), 20), col = "red", 
         main = "Keywords identified by RAKE", 
         xlab = "Rake")


# ## Using a sequence of POS tags (noun phrases / verb phrases)
# x$phrase_tag <- as_phrasemachine(x$upos, type = "upos")
# stats <- keywords_phrases(x = x$phrase_tag, term = tolower(x$token), 
#                           pattern = "(A|N)*N(P+D*(A|N)*N)*", 
#                           is_regex = TRUE, detailed = FALSE)
# stats <- subset(stats, ngram > 1 & freq > 3)
# stats$key <- factor(stats$keyword, levels = rev(stats$keyword))
# barchart(key ~ freq, data = head(stats, 20), col = "magenta", 
#          main = "Keywords - simple noun phrases", xlab = "Frequency")
# 




## N-gram charts {.tabset}

### Bigrams



library(dplyr) 
library(tidytext)

# Define bigram & trigram tokenizer 
tokenizer_bi <- function(x){
  NGramTokenizer(x, Weka_control(min=2, max=2))
}

tokenizer_tri <- function(x){
  NGramTokenizer(x, Weka_control(min=3, max=3))
}


# Text transformations
cleanVCorpus <- function(corpus){
  corpus.tmp <- tm_map(corpus, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, content_transformer(tolower))
  v_stopwords <- c(stopwords("en"),"will","etc","build","using")
  corpus.tmp <- tm_map(corpus.tmp, removeWords, v_stopwords)
  corpus.tmp <- tm_map(corpus.tmp, removeNumbers)
  return(corpus.tmp)
}

# Most frequent bigrams 
frequentBigrams <- function(text){
  s.cor <- VCorpus(VectorSource(text))
  s.cor.cl <- cleanVCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.cl, control=list(tokenize=tokenizer_bi))
  s.tdm <- removeSparseTerms(s.tdm, 0.999)
  m <- as.matrix(s.tdm)
  word_freqs <- sort(rowSums(m), decreasing=TRUE)
  dm <- data.frame(word=names(word_freqs), freq=word_freqs)
  return(dm)
}


# Most frequent bigrams
ep4.bigrams <- frequentBigrams(data.text$Description)[1:20,]
ggplot(data=ep4.bigrams, aes(x=reorder(word, -freq), y=freq)) +  
  geom_bar(stat="identity", fill="chocolate2", colour="black") +
  theme_bw() +
  theme(axis.text.x=element_text(angle=60, hjust=1)) +
  labs(x="Bigram", y="Frequency")




### Trigrams


# Most frequent trigrams 
frequentTrigrams <- function(text){
  s.cor <- VCorpus(VectorSource(text))
  s.cor.cl <- cleanVCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.cl, control=list(tokenize=tokenizer_tri))
  s.tdm <- removeSparseTerms(s.tdm, 0.999)
  m <- as.matrix(s.tdm)
  word_freqs <- sort(rowSums(m), decreasing=TRUE)
  dm <- data.frame(word=names(word_freqs), freq=word_freqs)
  return(dm)
}

# Most frequent trigrams
ep4.Trigrams <- frequentTrigrams(data.text$Description)[1:20,]
ggplot(data=ep4.Trigrams, aes(x=reorder(word, -freq), y=freq)) +  
  geom_bar(stat="identity", fill="midnightblue", colour="black") +
  theme_bw() +
  theme(axis.text.x=element_text(angle=60, hjust=1)) +
  labs(x="Trigram", y="Frequency")




## Key Data Science Tools


uniqueWords = function(text) {
text <- strsplit(text, " |,|/|;")
text <- lapply(text,unique)
text <- sapply(text, function(u) paste0(u, collapse = " "))
return(text) 
}

corpus = tm_map(corpus, content_transformer(uniqueWords))
corpustdm_all <- as.matrix(TermDocumentMatrix(corpus,control=list(wordLengths=c(1,Inf))))

freq <- rowSums(as.matrix(corpustdm_all))
#freq[ as.character(tolower(data.tools[,])) ]

m <- as.matrix(freq[ as.character(tolower(data.tools[,])) ])
n <- as.data.frame(rownames(m))
colnames(n) <- c("toolname")
n$frequency <- round(as.numeric(as.character(m[,1]))/ncol(corpustdm_all),2)
n <- na.omit(n)
#n <- subset(n,frequency>0.1)
n <- head(n[order(n$frequency, decreasing= T),], n = 20)

mycolors = colorRampPalette(brewer.pal(name="Blues", n = 12))(20)
#mycolors = c(brewer.pal(name="Dark2", n = 8), brewer.pal(name="Paired", n = 7))

ggplot(data=n, aes(x= reorder(n$toolname,-frequency), y=n$frequency, fill = n$toolname)) +
  geom_bar(stat="identity")+
  scale_y_continuous(labels = scales::percent) +
  geom_text(aes(label = paste0(frequency*100,"%")), 
  position = position_stack(vjust = 0.5), size = 3) +
  scale_color_manual(values = mycolors) +
  theme(legend.position="none") +
  labs(x = "Tools", y = "Percent") +
  ggtitle("Top 20 Tools in Data Scientist Job Listings") + 
  theme(plot.title = element_text(hjust=0.5)) +
  ggpubr::rotate_x_text()



