**Popular Natural Language Processing Text Preprocessing Techniques Implementation In Python**
Using the text preprocessing techniques we can remove noise from raw data and makes raw data more valuable for building models.

Here, raw data is nothing but data we collect from different sources like reviews from websites, documents, social media, twitter tweets, news articles etc.

Data preprocessing is the primary and most crucial step in any data science problems or project. Preprocessing the collected data is the integral part of any Natural Language Processing, Computer Vision, deep learning and machine learning problems. Based on the type of dataset, we have to follow different preprocessing methods.

Which means machine learning data preprocessing techniques vary from the deep learning, natural language or nlp data preprocessing techniques.

So there is a need to learn these techniques to build effective natural language processing models.

In this article we will discuss different text preprocessing techniques or methods like normalization, stemming, lemmatization, etc. for handling text to build various Natural Language Processing problems/models.

**Table of Contents**

Text Preprocessing Importance in NLP

Different Text Preprocessing Techniques

Converting to Lower case

Removal of HTML tags

Removal of URLs

Removing Numbers

Converting numbers to words

Apply spelling correction

Convert accented characters to ASCII characters

Expanding Contractions

Stemming

Lemmatization

Removal of Emojis

Removing of Punctuations or Special Characters

Removing of Frequent words

Removing of Rare words

Removing single characters

Removing Extra Whitespaces

Process of applying all text preprocessing techniques with an Example

**Text Preprocessing Importance in NLP**

we said before text preprocessing is the first step in the Natural Language Processing pipeline. The importance of preprocessing is increasing in NLP due to noise or unclear data extracted or collected from different sources.

Most of the text data collected from reviews of E-commerce websites like Amazon or Flipkart, tweets from twitter, comments from Facebook or Instagram, and other websites like Wikipedia, etc.

We can observe users use short forms, emojis, misspelling of words, etc. in their comments, tweets, and so on.

We should not feed raw data without preprocessing to build models because the preprocessing of text directly improves the model's performance.

If we feed data without performing any text preprocessing techniques, the build models will not learn the real significance of the data. In some cases, if we feed raw data without any preprocessing techniques the models will get confused and give random results.

In that confusion, the model will learn harmful patterns that are not valuable. Due to this, the model's performance will be affected, which means the model performance will reduce significantly.

So we should remove all these noises from the text and make it a more clear and structured form for building models.

Here we have to know one thing.

The natural language text preprocessing techniques will vary from problem to problem. This means we cannot apply the same text preprocessing techniques used for one NLP problem to another NLP problem.

For example, in sentiment analysis classification problems, we can remove or ignore numbers within the text because numbers are not significant in this problem statement.

However, we should not ignore the numbers if we are dealing with financial related problems. Because numbers play a key role in these kinds of problems.

So while performing NLP text preprocessing techniques. We need to focus more on the domain we are applying these NLP techniques and the order of methods also plays a key role.

Don't worry about the order of these techniques for now. We will give the generic order in which you need to apply these techniques.

Our suggestion is to use preprocessing methods or techniques on a subset of aggregate data (take a few sentences randomly). We can easily observe whether it is in our expected form or not. If it is in our expected form, then apply on a complete dataset; otherwise, change the order of preprocessing techniques.

We will provide a python file with a preprocess class of all preprocessing techniques at the end of this article.

You can download and import that class to your code. We can get preprocessed text by calling preprocess class with a list of sentences and sequences of preprocessing techniques we need to use.

Again the order of technique we need to use will differ from problem to problem.

**Different Text Preprocessing Techniques**

Let us jump to learn different types of text preprocessing techniques.

In the next few minutes, we will discuss and learn the importance and implementation of these techniques.

Converting to Lower case

Converting all our text into the lower case is a simple and most effective approach. If we are not applying lower case conversion on words like NLP, nlp, Nlp, we are treating all these words as different words.

After using the lower casing, all three words are treated as a single word that is nlp.!!

![download](https://user-images.githubusercontent.com/96867718/161251891-e15f3bb1-5b69-4f8b-9725-ffacf64b2e00.png)


This method is useful for problems that are dependent on the frequency of words such as document classification.

In this case, we count the frequency of words by using bag-of-words, TFIDF, etc.

It is better to perform lower case the text as the first step in this text preprocessing. Because if we are trying to remove stop words all words need to be in lower case.

For example, few sentences have the starting word as "The" if we are not performing the lower casing technique before that technique, we can not remove all stopwords.

The other case is for calculating the frequency count. If we not converted the text into lower case Data Science and data science will treat as different tokens.

In natural language processing the lower dimension of text which is words called as tokens.

We can apply this method to most of the text related problems. Still, it may not be suitable for different projects like Parts-Of-Speech tag recognition or dependency parsing, where proper word casing is essential to recognize nouns, verbs, etc.

Implementation of lower case conversion

# Implementation of lower case conversion

def lower_case_convertion(text):
    """
    Input :- string
    Output :- lowercase string
    """
    lower_text = text.lower()
    return lower_text


ex_lowercase = "This is an example Sentence for LOWER case conversion"
lowercase_result = lower_case_convertion(ex_lowercase)
print(lowercase_result)
this is an example sentence for lower case conversion

**Removal of HTML tags**

![download](https://user-images.githubusercontent.com/96867718/161251966-17d6f8be-e4e8-4e45-b32e-81ed569cd4b2.png)


This is the second essential preprocessing technique. The chances to get HTML tags in our text data is quite common when we are extracting or scraping data from different websites.

We don't get any valuable information from these HTML tags. So it is better to remove them from our text data. We can remove these tags by using regex and we can also use the BeautifulSoup module from bs4 libraries.

Let us see the implementation using python.

HTML tags removal Implementation using regex module

# HTML tags removal Implementation using regex module

import re
def remove_html_tags(text):
    """
    Return :- String without Html tags
    input :- String
    Output :- String
    """
    html_pattern = r'<.*?>'
    without_html = re.sub(pattern=html_pattern, repl=' ', string=text)
    return without_html

ex_htmltags = """ <body>
<div>
<h1>Hi, this is an example text with Html tags. </h1>
</div>
</body>
"""
htmltags_result = remove_html_tags(ex_htmltags)
print(f"Result :- \n {htmltags_result}")
Result :- 
   
 
 Hi, this is an example text with Html tags.  
 
 ![download](https://user-images.githubusercontent.com/96867718/161252020-07e1d6cf-c18a-493c-8b3e-ea8932a6d1c4.png)



Implementation of Removing HTML tags using bs4 library

# Implementation of Removing HTML tags using bs4 library

from bs4 import BeautifulSoup
def remove_html_tags_beautifulsoup(text):
	"""
	Return :- String without Html tags
	input :- String
	Output :- String
	"""
	parser = BeautifulSoup(text, "html.parser")
	without_html = parser.get_text(separator = " ")
	return without_html

ex_htmltags = """ <body>
<div>
<h1>Hi, this is an example text with Html tags. </h1>
</div>
</body>
"""
htmltags_result = remove_html_tags_beautifulsoup(ex_htmltags)
print(f"Result :- \n {htmltags_result}")
Result :- 
   
 
 Hi, this is an example text with Html tags.  
 
 

We can observe both the functions are giving the same result after removing HTML tags from our example text.

**Removal of URLs**
![download](https://user-images.githubusercontent.com/96867718/161252105-e44f695e-ad36-409d-af5d-d79351e08861.png)


URL is the short-form of Uniform Resource Locator. The URLs within the text refer to the location of another website or anything else.

If we are performing any website backlinks analysis, twitter or Facebook in that case, URLs are an excellent choice to keep in text.

Otherwise, from URLs also we can not get any information. So we can remove it from our text. We can remove URLs from the text by using the python Regex library.

![download](https://user-images.githubusercontent.com/96867718/161252127-87330c0b-d875-4ab5-9ec2-e23a1ba1b41d.png)



# Implementation of Removing URLs using python regex

In the below script. We take example text with URLs and then call the 2 functions with that example text. In the remove_urls function, assign a regular expression to remove URLs to url_pattern after That, substitute URLs within the text with space by calling the re library's sub-function.

# Implementation of Removing URLs  using python regex

import re
def remove_urls(text):
    """
    Return :- String without URLs
    input :- String
    Output :- String
    """
    url_pattern = r'https?://\S+|www\.\S+'
    without_urls = re.sub(pattern=url_pattern, repl=' ', string=text)
    return without_urls

# example text which contain URLs in it
ex_urls = """
This is an example text for URLs like http://google.com & https://www.facebook.com/ etc.
"""

# calling removing_urls function with example text (ex_urls)
urls_result = remove_urls(ex_urls)
print(f"Result after removing URLs from text :- \n {urls_result}")
Result after removing URLs from text :- 
 
This is an example text for URLs like   &   etc.

# Removing Numbers

We can remove numbers from the text if our problem statement doesn't require numbers.

For example, if we are working on financial related problems like banking or insurance-related sectors. We may get information from numbers.

In those cases, we shouldn't remove numbers.

![download](https://user-images.githubusercontent.com/96867718/161252208-09092a2f-7e2c-48df-8d0e-13c29d47a2b8.png)


Implementation of Removing numbers using python regex

In the code below, we will call the remove_numbers function with example text, which contains numbers.

Let's see how to implement it.

# Implementation of Removing numbers  using python regex

import re
def remove_numbers(text):
    """
    Return :- String without numbers
    input :- String
    Output :- String
    """
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern,
    repl=" ", string=text)
    return without_number

# example text which contain numbers in it
ex_numbers = """
This is an example sentence for removing numbers like 1, 5,7, 4 ,77 etc.
"""
# calling remove_numbers function with example text (ex_numbers)
numbers_result = remove_numbers(ex_numbers)
print(f"Result after removing number from text :- \n {numbers_result}")
Result after removing number from text :- 
 
This is an example sentence for removing numbers like  ,  , ,   ,  etc.

In the above removing_numbers function. We mentioned a pattern to recognize numbers within the text and then substitute numbers with space using the re library's sub-function.

And then return text after removing the number to numbers_result variable.

# Converting numbers to words

our problem statement need valuable information from numbers in that case, we have to convert numbers to words. Similar problem statements which are discussed at the removing numbers (above section).

![download](https://user-images.githubusercontent.com/96867718/161252274-f2f74881-a33f-496c-b4f1-7d7abe1727d5.png)


Implementation of Converting numbers to words using python num2words library

We can convert numbers to words by just importing the num2words library. In the code below, we will call the num_to_words function with example text. Example text has numbers.

pip install num2words
Requirement already satisfied: num2words in /home/santhosh/anaconda3/lib/python3.9/site-packages (0.5.10)
Requirement already satisfied: docopt>=0.6.2 in /home/santhosh/anaconda3/lib/python3.9/site-packages (from num2words) (0.6.2)
Note: you may need to restart the kernel to use updated packages.
# function to convert numbers to words
from num2words import num2words

def num_to_words(text):
    """
    Return :- text which have all numbers or integers in the form of words
    Input :- string
    Output :- string
    """
    # splitting text into words with space
    after_spliting = text.split()

    for index in range(len(after_spliting)):
        if after_spliting[index].isdigit():
            after_spliting[index] = num2words(after_spliting[index],lang='en')

    # joining list into string with space
    numbers_to_words = ' '.join(after_spliting)
    return numbers_to_words

# example text which contain numbers in it
ex_numbers = "This is an example sentence for converting numbers to words like 1 to one, 5 to five, 74 to seventy-four, etc."
# calling remove_numbers function with example text (ex_numbers)
numners_result = num_to_words(ex_numbers)
print(f"Result after converting numbers to its words from text :- \n {numners_result}")

## Output:: This is an example sentence for converting numbers to words like one to one, five to five, seventy-four to seventy-four, etc.
Result after converting numbers to its words from text :- 
 This is an example sentence for converting numbers to words like one to one, five to five, seventy-four to seventy-four, etc.
In the above code, the num_to_words function is getting the text as input. In that, we are splitting text using a python string function of a split with space to get words individually.

Taking each word and checking if that word is digit or not. If the word is digit then convert that into words.

# Apply spelling correction

![download](https://user-images.githubusercontent.com/96867718/161252336-a0dec74e-7ed5-4f9a-ae47-3802297f64a9.png)


Spelling correction is another important preprocessing technique while working with tweets, comments, etc. Because we can see incorrect spelling words in those areas of text. We need to make those misspelling words to correct spelling words.

We can check and replace misspelling words with correct spelling by using two python libraries, one is pyspellchecker, and another one is autocorrect.

![download](https://user-images.githubusercontent.com/96867718/161252388-457be9c9-0869-430c-a42c-560573408bf8.png)



Implementation of spelling correction using python pyspellchecker library

Below we are calling a spell_correction function with example text. Example text has incorrect spelling words to check whether the spell_correction function gives correct words or not.

pip install pyspellchecker
Requirement already satisfied: pyspellchecker in /home/santhosh/anaconda3/lib/python3.9/site-packages (0.6.3)
Note: you may need to restart the kernel to use updated packages.
# Implementation of spelling correction using python pyspellchecker library

from spellchecker import SpellChecker

spell_corrector = SpellChecker()

# spelling correction using spellchecker
def spell_correction(text):
    """
    Return :- text which have correct spelling words
    Input :- string
    Output :- string
    """
    # initialize empty list to save correct spell words
    correct_words = []
    # extract spelling incorrect words by using unknown function of spellchecker
    misSpelled_words = spell_corrector.unknown(text.split())

    for each_word in text.split():
        if each_word in misSpelled_words:
            right_word = spell_corrector.correction(each_word)
            correct_words.append(right_word)
        else:
            correct_words.append(each_word)

    # joining correct_words list into single string
    correct_spelling = ' '.join(correct_words)
    return correct_spelling

#example text with mis spelling words
ex_misSpell_words = """
This is an example sentence for spell corecton
"""
spell_result = spell_correction(ex_misSpell_words)
print(f"Result after spell checking :- \n{spell_result}")
Result after spell checking :- 
This is an example sentence for spell correction
Implementation of spelling correction using python autocorrect library

pip install autocorrect
Requirement already satisfied: autocorrect in /home/santhosh/anaconda3/lib/python3.9/site-packages (2.6.1)
Note: you may need to restart the kernel to use updated packages.
pip install nltk
Requirement already satisfied: nltk in /home/santhosh/anaconda3/lib/python3.9/site-packages (3.2.4)
Requirement already satisfied: six in /home/santhosh/anaconda3/lib/python3.9/site-packages (from nltk) (1.16.0)
Note: you may need to restart the kernel to use updated packages.
# Implementation of spelling correction using python autocorrect library
import nltk
nltk.download()

from autocorrect import Speller
from nltk import word_tokenize

# spelling correction using spellchecker
def spell_autocorrect(text):
    """
    Return :- text which have correct spelling words
    Input :- string
    Output :- string
    """
    correct_spell_words = []

    # initialize Speller object for english language with 'en'
    spell_corrector = Speller(lang='en')
    for word in word_tokenize(text):
        # correct spell word
        correct_word = spell_corrector(word)
        correct_spell_words.append(correct_word)

    correct_spelling = ' '.join(correct_spell_words)
    return correct_spelling

# another example text with misSpelling words
ex_misSpell_words_1 = """
This is anoter exapl for spell correction
"""
spell_result = spell_autocorrect(ex_misSpell_words_1)
print(f"Result :- \n{spell_result}")
showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
Result :- 
This is another example for spell correction
We can observe both methods given correct or expected solutions.

# We can observe both methods given correct or expected solutions.
This is another common preprocessing technique in NLP. We can observe special characters at the top of the common letter or characters if we press a longtime while typing, for example, résumé.

If we are not removing these types of noise from the text, then the model will consider resume and résumé; both are two different words.

Even if both are the same. We can convert this accented character to ASCII characters by using the unidecode library.

![download](https://user-images.githubusercontent.com/96867718/161252509-b0179823-60ea-4931-b7cc-0128e452dbd8.png)


Implementation of accented text to ASCII converter in python

We will define the accented_to_ascii function to convert accented characters to their ASCII values in the below script.

We will do this function with example text.

# Implementation of accented text to ASCII converter in python

import unidecode

def accented_to_ascii(text):
    """
    Return :- text after converting accented characters
    Input :- string
    Output :- string
    """
    # apply unidecode function on text to convert
    # accented characters to ASCII values
    text = unidecode.unidecode(text)
    return text

# example text with accented characters
ex_accented = """
This is an example text with accented characters like dèèp lèarning ánd cömputer vísíön etc.
"""
accented_result = accented_to_ascii(ex_accented)
print(f"Result after converting accented characters to their ASCII values \n{accented_result}")
Result after converting accented characters to their ASCII values 

This is an example text with accented characters like deep learning and computer vision etc.

The above code, we use the unidecode method of the unidecode library with input text. Which converts accented characters to ASCII values

# Stemming

![download](https://user-images.githubusercontent.com/96867718/161252556-c72afea3-d3d3-47f9-b36a-330afbe6ab88.png)

Stemming is reducing words to their base or root form by removing a few suffix characters from words. Stemming is the text normalization technique.

There are so many stemming algorithms available, but the most widely used one is porter stemming.

For example, the result of books after stemming is a book, and the result of learning is learn.

![download](https://user-images.githubusercontent.com/96867718/161252589-767b5917-16dd-4f23-b644-6c872c794826.png)


But stemming doesn't always provide the correct form of words because this follows the rules like removing suffix characters to get base words.

Sometimes, stemming words don't relate to original ones and sometimes give non - dictionary words or not proper words.

For this, we can observe in the above table results of stemming "caring" and "console/consoling". Because of these results stemming technique does not apply to all NLP tasks.

Implementation of Stemming using PorterStemming from nltk library In the below python script, we will define the porter_stemmer function to implement the stemming technique. We will call the function with example text.

Before reaching the function, we have to initialize the object for the PorterStemmer class to use the stem function from that class.

Implementation of Stemming using PorterStemming from nltk library

In the below python script, we will define the porter_stemmer function to implement the stemming technique. We will call the function with example text.

Before reaching the function, we have to initialize the object for the PorterStemmer class to use the stem function from that class.

# Implementation of Stemming using PorterStemming from nltk library

from nltk.stem import PorterStemmer

def porter_stemmer(text):
	"""
	Result :- string after stemming
	Input :- String
	Output :- String
	"""
	# word tokenization
	tokens = word_tokenize(text)

	for index in range(len(tokens)):
		# stem word to each word
		stem_word = stemmer.stem(tokens[index])
		# update tokens list with stem word
		tokens[index] = stem_word

	# join list with space separator as string
	return ' '.join(tokens)

# initialize porter stemmer object
stemmer = PorterStemmer()
# example text for stemming technique
ex_stem = "Programers program with programing languages"
stem_result = porter_stemmer(ex_stem)
print(f"Result after stemming technique :- \n{stem_result}")
Result after stemming technique :- 
program program with program languag
In the porter_stemmer function, we tokenized the input using word_tokenize from the nltk library. And then, apply the stem function to each of the tokenized words and update the text with stemmer words.

# Lemmatization

![download](https://user-images.githubusercontent.com/96867718/161252640-1233f204-0481-49dd-b6dd-47f3b29d6260.png)

The aim of usage of lemmatization is similar to the stemming technique to reduce inflection words to their original or base words. But the lemmatization process is different from the above approach.

Lemmatization does not only trim the suffix characters; instead, use lexical knowledge bases to get original words. The result of lemmatization is always a meaningful word, not like stemming.

The disadvantages of stemming people prefer to use lemmatization to get base or root words of original words. This preprocessing technique is also optional; we have to apply it based on our problem statement.

Suppose we are doing POS (parts-of-speech) tagger problems. The original words of data have more information about data. As compared to stemming, the lemmatization speed is a little bit slow.

Let's see the implementation of lemmatization using nltk library.

Implementation of lemmatization using nltk

In the below strip, before calling the lemmatization function, we have to initialize the object for WordNetLemmatizer to use it.

## Implementation of lemmatization using nltk

from nltk.stem import WordNetLemmatizer

def lemmatization(text):
	"""
	Result :- string after stemming
	Input :- String
	Output :- String
	"""
	# word tokenization
	tokens = word_tokenize(text)

	for index in range(len(tokens)):
		# lemma word
		lemma_word = lemma.lemmatize(tokens[index])
		tokens[index] = lemma_word

	return ' '.join(tokens)

# initialize lemmatizer object
lemma = WordNetLemmatizer()
# example text for lemmatization
ex_lemma = """
Programers program with programing languages
"""
lemma_result = lemmatization(ex_lemma)
print(f"Result of lemmatization \n{lemma_result}")
Result of lemmatization 
Programers program with programing language
We can see the differences between the outputs of stemming and lemmatization. Programmers program programming all are different, and for languages, lemma gives meaningful words but stemming words for that are meaningless.

# Differences between Stemming and Lemmatization
![download](https://user-images.githubusercontent.com/96867718/161252683-73a3558a-60aa-43ed-9e68-e94d0f4ae641.png)


# Removal of Emojis

![download](https://user-images.githubusercontent.com/96867718/161252705-8a5b623b-66a3-4039-8f0b-512635684df7.png)


In today's online communication, emojis play a very crucial role.

Emojis are small images. Users use these emojis to express their present feelings. We can communicate these with anyone globally. For some problem statements, we need to remove emojis from the text.

Let's see on that type of problem statement how we can remove emojis.

Implementation of emoji removing For this we take code snippets from this GitHub Repo.

# Implementation of emoji removing
![download](https://user-images.githubusercontent.com/96867718/161252764-d73dd89d-ce20-4491-825e-efac85057a76.png)



# Implementation of emoji removing

def remove_emojis(text):
	"""
	Result :- string without any emojis in it
	Input :- String
	Output :- String
	"""
	emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)

	without_emoji = emoji_pattern.sub(r'',text)
	return without_emoji


# example text for emoji removing technique
ex_emoji = """
This is a test 😻 👍🏿
"""
# calling function
emoji_result = remove_emojis(ex_emoji)
print(f"Result text after removing emojis :- \n{emoji_result}")
Result text after removing emojis :- 

This is a test  

# Removing of Punctuations or Special Characters

![download](https://user-images.githubusercontent.com/96867718/161252818-b5441348-c399-4313-8cae-80c383d1ecad.png)


Punctuations or special characters are all characters except digits and alphabets. List of all available special characters are [!"#$%&'()*+,-./:;<=>?@[]^_`{|}~].

This is better to remove or convert emoticons before removing punctuations or special characters.

If we apply this technique process before emoticons related techniques, we may lose emoticons from the text. So if we apply the emoticons technique, apply before removing the punctuation technique.

For example, if we remove the period using the punctuation removing technique from text like "money 20.98", we will lose the period (.) between 20 & 98. That completely lost their meaning.

So we have to focus more on choosing punctuations.

![download](https://user-images.githubusercontent.com/96867718/161252862-e298611e-5953-44e3-b8e6-16bb176c8eef.png)


Implementation of removing punctuations using string library

# Implementation of removing punctuations using string library

from string import punctuation

def remove_punctuation(text):
	"""
	Return :- String after removing punctuations
	Input :- String
	Output :- String
	"""
	return text.translate(str.maketrans('', '', punctuation))


# example text for removing punctuations
ex_punct = """
this is an example text for punctuations like .?/*
"""
punct_result = remove_punctuation(ex_punct)
print(f"Result after removing punctuations :- \n{punct_result}")
Result after removing punctuations :- 

this is an example text for punctuations like 

Removing of Frequent words

The above section, we removed stopwords.

Stopwords are common words all over the language. These frequent words are common words of a particular domain.

If we are working on any problem statement for a specific field, we can ignore common words in that domain because those frequent words don't give too much information.

Implementation of frequent words removing

Here we use the "Counter" function from the collection library to remove our corpus's frequent words.

## Implementation of frequent words removing

from collections import Counter

def freq_words(text):
    """
    Return :- Most frequent words
    Input :- string
    Output :-
    """
    # tokenization
    tokens = word_tokenize(text)
    for word in tokens:
        counter[word]= +1

    FrequentWords = []
    # take top 10 frequent words
    for (word, word_count) in counter.most_common(10):
        FrequentWords.append(word)

    return FrequentWords

def remove_fw(text, FrequentWords):
    """
    Return :- String after removing frequent words
    Input :- String
    Output :- String
    """

    tokens = word_tokenize(text)
    without_fw = []
    for word in tokens:
        if word not in FrequentWords:
            without_fw.append(word)

    without_fw = ' '.join(without_fw)
    return without_fw


# initiate object for counter
counter = Counter()
# some random text on machine learning
ex_fw = """
Machine learning is the idea that there are generic algorithms that can tell you something interesting about a set of data without you having to write any custom code specific to the problem. Instead of writing code, you feed data to the generic algorithm and it builds its own logic based on the data.
For example, one kind of algorithm is a classification algorithm. It can put data into different groups. The same classification algorithm used to recognize handwritten numbers could also be used to classify emails into spam and not-spam without changing a line of code. It's the same algorithm but it's fed different training data so it comes up with different classification logic.
Two kinds of Machine Learning Algorithms
You can think of machine learning algorithms as falling into one of two main categories -- supervised learning and unsupervised learning. The difference is simple, but really important.
Supervised Learning
Let's say you are a real estate agent. Your business is growing, so you hire a bunch of new trainee agents to help you out. But there's a problem -- you can glance at a house and have a pretty good idea of what a house is worth, but your trainees don't have your experience so they don't know how to price their houses.
To help your trainees (and maybe free yourself up for a vacation), you decide to write a little app that can estimate the value of a house in your area based on it's size, neighborhood, etc, and what similar houses have sold for.
So you write down every time someone sells a house in your city for 3 months. For each house, you write down a bunch of details -- number of bedrooms, size in square feet, neighborhood, etc. But most importantly, you write down the final sale price:
This is called supervised learning. You knew how much each house sold for, so in other words, you knew the answer to the problem and could work backwards from there to figure out the logic.
To build your app, you feed your training data about each house into your machine learning algorithm. The algorithm is trying to figure out what kind of math needs to be done to make the numbers work out.
This kind of like having the answer key to a math test with all the arithmetic symbols erased:
"""

# calling count_fw to calculate frequent words
FrequentWords = freq_words(ex_fw)
print(f"Top 10 Frequent Words from our example text :- \n{FrequentWords}")


# calling remove_fw to remove frequent words from example text
fw_result = remove_fw(ex_fw, FrequentWords)

print(f"Result after removing frequent words :-\n{fw_result}")
Top 10 Frequent Words from our example text :- 
['Machine', 'learning', 'is', 'the', 'idea', 'that', 'there', 'are', 'generic', 'algorithms']
Result after removing frequent words :-
can tell you something interesting about a set of data without you having to write any custom code specific to problem . Instead of writing code , you feed data to algorithm and it builds its own logic based on data . For example , one kind of algorithm a classification algorithm . It can put data into different groups . The same classification algorithm used to recognize handwritten numbers could also be used to classify emails into spam and not-spam without changing a line of code . It 's same algorithm but it 's fed different training data so it comes up with different classification logic . Two kinds of Learning Algorithms You can think of machine as falling into one of two main categories -- supervised and unsupervised . The difference simple , but really important . Supervised Learning Let 's say you a real estate agent . Your business growing , so you hire a bunch of new trainee agents to help you out . But 's a problem -- you can glance at a house and have a pretty good of what a house worth , but your trainees do n't have your experience so they do n't know how to price their houses . To help your trainees ( and maybe free yourself up for a vacation ) , you decide to write a little app can estimate value of a house in your area based on it 's size , neighborhood , etc , and what similar houses have sold for . So you write down every time someone sells a house in your city for 3 months . For each house , you write down a bunch of details -- number of bedrooms , size in square feet , neighborhood , etc . But most importantly , you write down final sale price : This called supervised . You knew how much each house sold for , so in other words , you knew answer to problem and could work backwards from to figure out logic . To build your app , you feed your training data about each house into your machine algorithm . The algorithm trying to figure out what kind of math needs to be done to make numbers work out . This kind of like having answer key to a math test with all arithmetic symbols erased :
The above script, we defined two functions one is for counting frequent words another is to remove them from our corpus.

Removing of Rare words

Removing rare words text preprocessing technique is similar to eliminating frequent words. We can remove more irregular words from the corpus.

Implementation of frequent words removing

In the below script, the same as the above one, we defined two functions: finding rare words and removing them. We take only ten rare words for this sample text; this number may increase based on our text corpus.

# Implementation of rare words removing

from collections import Counter

def rare_words(text):
    """
    Return :- Most Rare words
    Input :- string
    Output :- list of rare words
    """
    # tokenization
    tokens = word_tokenize(text)
    for word in tokens:
        counter[word]= +1

    RareWords = []
    number_rare_words = 10
    # take top 10 frequent words
    frequentWords = counter.most_common()
    for (word, word_count) in frequentWords[:-number_rare_words:-1]:
        RareWords.append(word)

    return RareWords

def remove_rw(text, RareWords):
    """
    Return :- String after removing frequent words
    Input :- String
    Output :- String
    """

    tokens = word_tokenize(text)
    without_rw = []
    for word in tokens:
        if word not in RareWords:
            without_rw.append(word)

    without_rw = ' '.join(without_fw)
    return without_rw
# initiate object for counter
counter = Counter()
# some random text on machine learning
ex_fw = """
Machine learning is the idea that there are generic algorithms that can tell you something interesting about a set of data without you having to write any custom code specific to the problem. Instead of writing code, you feed data to the generic algorithm and it builds its own logic based on the data.
For example, one kind of algorithm is a classification algorithm. It can put data into different groups. The same classification algorithm used to recognize handwritten numbers could also be used to classify emails into spam and not-spam without changing a line of code. It's the same algorithm but it's fed different training data so it comes up with different classification logic.
Two kinds of Machine Learning Algorithms
You can think of machine learning algorithms as falling into one of two main categories -- supervised learning and unsupervised learning. The difference is simple, but really important.
Supervised Learning
Let's say you are a real estate agent. Your business is growing, so you hire a bunch of new trainee agents to help you out. But there's a problem -- you can glance at a house and have a pretty good idea of what a house is worth, but your trainees don't have your experience so they don't know how to price their houses.
To help your trainees (and maybe free yourself up for a vacation), you decide to write a little app that can estimate the value of a house in your area based on it's size, neighborhood, etc, and what similar houses have sold for.
So you write down every time someone sells a house in your city for 3 months. For each house, you write down a bunch of details -- number of bedrooms, size in square feet, neighborhood, etc. But most importantly, you write down the final sale price:
This is called supervised learning. You knew how much each house sold for, so in other words, you knew the answer to the problem and could work backwards from there to figure out the logic.
To build your app, you feed your training data about each house into your machine learning algorithm. The algorithm is trying to figure out what kind of math needs to be done to make the numbers work out.
This kind of like having the answer key to a math test with all the arithmetic symbols erased:
"""
# calling rare_words to calculate rare words
RareWords = rare_words(ex_fw)
print(f"Top 10 Rarer Words from our example text :- \n{RareWords}\n")

# calling remove_fw to remove rare words from example text
rw_result = remove_fw(ex_fw, RareWords)

print(f"Result after removing rare words :-\n{rw_result}")
Top 10 Rarer Words from our example text :- 
['erased', 'symbols', 'arithmetic', 'all', 'test', 'key', 'like', 'make', 'done']

Result after removing rare words :-
Machine learning is the idea that there are generic algorithms that can tell you something interesting about a set of data without you having to write any custom code specific to the problem . Instead of writing code , you feed data to the generic algorithm and it builds its own logic based on the data . For example , one kind of algorithm is a classification algorithm . It can put data into different groups . The same classification algorithm used to recognize handwritten numbers could also be used to classify emails into spam and not-spam without changing a line of code . It 's the same algorithm but it 's fed different training data so it comes up with different classification logic . Two kinds of Machine Learning Algorithms You can think of machine learning algorithms as falling into one of two main categories -- supervised learning and unsupervised learning . The difference is simple , but really important . Supervised Learning Let 's say you are a real estate agent . Your business is growing , so you hire a bunch of new trainee agents to help you out . But there 's a problem -- you can glance at a house and have a pretty good idea of what a house is worth , but your trainees do n't have your experience so they do n't know how to price their houses . To help your trainees ( and maybe free yourself up for a vacation ) , you decide to write a little app that can estimate the value of a house in your area based on it 's size , neighborhood , etc , and what similar houses have sold for . So you write down every time someone sells a house in your city for 3 months . For each house , you write down a bunch of details -- number of bedrooms , size in square feet , neighborhood , etc . But most importantly , you write down the final sale price : This is called supervised learning . You knew how much each house sold for , so in other words , you knew the answer to the problem and could work backwards from there to figure out the logic . To build your app , you feed your training data about each house into your machine learning algorithm . The algorithm is trying to figure out what kind of math needs to be to the numbers work out . This kind of having the answer to a math with the :
Removing single characters

After performing all text preprocessing techniques except extra spaces, removing this is better to remove a single character if there is any present in our corpus. We can remove using regex.

Implementation of removing single characters

## Remove single characters

def remove_single_char(text):
    """
    Return :- string after removing single characters
    Input :- string
    Output:- string
    """
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc

# example text for removing single characters
ex_sc = """
this is an example of single characters like a , b , and c .
"""
# calling remove_sc function to remove single characters
sc_result = remove_single_char(ex_sc)
print(f"Result :-\n{sc_result}")
Result :-

this is an example of single characters like , , and .

Removing Extra Whitespaces

This is the last preprocessing technique. We can not get any information from extra spaces, so that we can ignore all additional spaces such as 0ne or more newlines, tabs, extra spaces.

Our suggestion is to apply this preprocessing technique at last after performing all text preprocessing techniques.

Implementation of removing extra whitespaces

# Removing Extra Whitespaces

import re

def remove_extra_spaces(text):
	"""
	Return :- string after removing extra whitespaces
	Input :- String
	Output :- String
	"""
	space_pattern = r'\s+'
	without_space = re.sub(pattern=space_pattern, repl=" ", string=text)
	return without_space


# example text for removing extra spaces
ex_space = """
this      is an
extra spaces        .
"""

space_result = remove_extra_spaces(ex_space)
print(f"Result :- \n{space_result}")
Result :- 
 this is an extra spaces . 
Process of applying all text preprocessing techniques with an Example

For this process, we are providing a complete python code in our dataaspirant github repo. You have to download this preprocessing.py file After extracting the downloaded file.

Import it into our text preprocessing class from the preprocessing file. Now we will discuss how to use it.

# Implementation of Complete preprocessing techniques

![download](https://user-images.githubusercontent.com/96867718/161252981-6eb26864-f80f-45cd-80eb-222e7bbe62dd.png)


The below, we apply only a few text preprocessing techniques to know how we can use the importing class.

Here we are taking the Sms_spam_or_not dataset.

From the dataset, we are taking a text column and converting it into a list. We initiated an object for the prepress class, which one imported from a preprocessing file.

If we want to apply preprocessing techniques, send a list of sentences and a list of techniques to the preprocessing function by using the object of preprocessing.

We listed out all techniques with short forms in the comment section. Please send a list of short forms of corresponding techniques as a technique list.

 
 
