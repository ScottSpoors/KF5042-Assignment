emb = fastTextWordEmbedding; %load the pre-trained word embedding

data = readLexicon; %read the lexicon provided with the dataset

idx = ~isVocabularyWord(emb,data.Word);%checks if words contained in file are within the word embedding
data(idx,:) = [];

numWords = size(data,1);
cvp = cvpartition(numWords,'HoldOut',0.1); %split data into testing and training sections
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);

wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain); %converts training words to vectors
YTrain = dataTrain.Label;

mdl = fitcsvm(XTrain,YTrain);

wordsTest = dataTest.Word;
XTest = word2vec(emb,wordsTest); %converts testing words to vectors
YTest = dataTest.Label;

[YPred,scores] = predict(mdl,XTest); %predicts sentiment scores

filename = "labeled_data.csv";
tbl = readtable(filename,'TextType','string'); 
textData = tbl.tweet; %reads the data and selects specifically the column containing tweets

documents = preprocessText(textData); %calls the function to process the text to ready it for analysis
idx = ~isVocabularyWord(emb,documents.Vocabulary); %checks the words in the dataset against the word embedding
documents = removeWords(documents,idx); %removes words from the dataset not contained in the embedding



for i = 1:numel(documents) %loops through all words in data and predicts sentiment score
   words = string(documents(i));
   vec = word2vec(emb,words);
   [~,scores] = predict(mdl,vec);
   sentimentScore(i) = mean(scores(:,1));
end

table(sentimentScore',textData) %generates a table of all of the tweets with accompanying sentiment scores

function data = readLexicon %reads the lexicon provided within the dataset
    fidHateSpeech = fopen(fullfile('Dataset','hate-speech-and-offensive-language-master','lexicons','refined_ngram_dict.csv'));
    C = textscan(fidHateSpeech,'%s','CommentStyle',';');
    HateSpeech = string(C{1});
    
    fclose all;
    
    words = [HateSpeech];
    labels = categorical(nan(numel(words),1));
    labels(1:numel(HateSpeech)) = "HateSpeech"; %adds labels to the text data
    
    data = table(words,labels,'VariableNames',{'Word','Label'}); %creates a table of text with accompanying labels
end

function documents = preprocessText(textData)
    documents = tokenizedDocument(textData); %splits tweets into individual words
    documents = erasePunctuation(documents); %removes all punctuation from tweets
    documents = removeStopWords(documents); %removes unnecessary words e.g. 'and', 'but', and 'the'
    documents = lower(documents); %converts all words to lower case
end
