# Get To The Point: Summarization with Pointer-Generator Networks
Pytorch implementation of [Get To The Point: Summarization with Pointer-Generator Networks (2017)](https://arxiv.org/pdf/1704.04368.pdf) by Abigail See et al. 

## Model Description
* LSTM based Sequence-to-Sequence model for Abstractive Summarization
* Pointer mechanism for handling Out of Vocabulary (OOV) words [See et al. (2017)](https://arxiv.org/pdf/1704.04368.pdf)
* FastText used for creating embeddings over the dataset .

## Model Architecture

<p align="center">
<img src="https://github.com/Developer-Zer0/Text-Summarization/blob/master/Assets/model_architecture.png">
</p>

## Prerequisites
* Pytorch
* gensim
* python 3

## Data
Data set used : [Kaggle:Amazon fine food reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) 


# Examples
```Sentence:``` once more amazon was great the product is good for kids even though it has a little bit more sugar than needed</br> 
```Predicted Summary:``` <START> good as expected <END></br>
```Actual Summary:``` <START> as expected <END></br>

```Sentence:``` this is an excellent tea for a breakfast tea or for the afternoon or evening it has a wonderful mellow flavor in the morning i like to brew it with earl grey to create a nice smooth blend it is a great way to start the day</br>
```Predicted Summary:``` <START> great product tea <END></br>
```Actual Summary:``` <START> wonderful anytime tea <END></br>

```Sentence:``` very tasty this mix makes a moist yummy cake we make our own cream cheese frosting with the chocolate cake and it is a winner</br>
```Predicted Summary:```<START> tasty <END> </br>
```Actual Summary:``` <START> versatile and yummy <END> </br>

```Sentence:``` ca not complain taste good and had quick delivery it was my first time trying this tea out i usually drink the peppermint one but this gave me energy and sustained me throughout the day </br>
```Predicted Summary:``` <START> very good good <END></br>
```Actual Summary:``` <START> very good <END></br>

```Sentence:``` energy bites f ing rip your face off with molten lava energy br br infinity energy to the f ing max i ate a whole box once and threw a car at a baby br br f ing rave br br seriously these are great</br>
```Predicted Summary:``` <START> love it <END></br>
```Actual Summary:``` <START> maximum rave power <END></br>

```Sentence:``` looks good from the package but does not taste good at all hard crunchy freeze dried flavor br save your money guys</br>
```Predicted Summary:``` <START> yuck <END></br>
```Actual Summary:``` <START> yuck <END></br>

```Sentence:``` this is the best dog food their is because everything is very digestible and when the dog does digest it all digested because they use all the nutrition from it so it is healther and the best out their far as i am think </br>
```Predicted Summary:``` <START> best dog food <END> </br>
```Actual Summary:```  <START> natures logic venisen <END> </br>

```Sentence:``` great coffee br excellent service br best way to buy k cups br stock up before coffee prices go up again </br>
```Predicted Summary:``` <START> great coffee <END> </br>
```Actual Summary:``` <START> great coffee excellent service <END> </br>

```Sentence:``` wonderful flavor would purchase this blend of coffee again light flavor not bitter at all and price was great the best i found anywhere </br>
```Predicted Summary:``` <START> great flavor <END> </br>
```Actual Summary:``` <START> wolfgang puck k cup breakfast in bed <END> </br>

```Sentence:``` no wonder they were so cheap they do not work very well in my machine because of the biodegradable packaging i would not buy them again </br>
```Predicted Summary:``` <START> disappointed <END></br>
```Actual Summary:``` <START> ethical coffee nespresso capsules <END> </br>

```Sentence:``` i bought these from a large chain pet store after reading the reviews i checked the bag made in china i threw the whole bag away i wish i would have read the reviews first </br>
```Predicted Summary:```  <START> do not buy <END> </br>
```Actual Summary:``` <START> do not buy <END> </br>

```Sentence:``` i love these gums they are not as cloyingly sweet as american gummies and have a lot more fruit flavor the only problem is that i ca not eat just one </br>
```Predicted Summary:``` <START> a addictive <END> </br>
```Actual Summary:```  <START> danger highly addictive <END> </br>

```Sentence:```  the pepper plant habanero extra hot california style hot pepper sauce 10 oz has great flavor as all the pepper plants do i just love it it is a bit pricey but worth it </br>
```Predicted Summary:``` <START> great seasoning <END> </br>
```Actual Summary:``` <START> wonderful love it <END> </br>

------------------------------------------

<h3 align='center'>Loss for 75k iterations</h3></br>

<p align="center">
<img src="https://github.com/Developer-Zer0/Text-Summarization/blob/master/Assets/Loss_graph.jpeg">
</p>


------------------------------------------
## Contributors:

- [Ankur Chemburkar](https://github.com/Developer-Zer0)
- [Talha Chafekar](https://github.com/talha1503)

------------------------------------------
## References
* [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
