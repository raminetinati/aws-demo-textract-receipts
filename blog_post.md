# Amazon Textract - Building a Receipt Processing Solution

## Overview

In [May 2019](https://press.aboutamazon.com/news-releases/news-release-details/aws-announces-general-availability-amazon-textract) Amazon announced the General Availability of Amnazon Textract, which is a fully managed service that uses machine learning to automatically extract text and data, including from tables and forms, in virtually any document – with no machine learning experience required. In this blog post, we"re going to explore the use of Amazon Textract to build a Receipt Processing Solution, which uses images of different types of receipts, and demonstrates how to apply different methods to process the data.

For this solution, the [code](https://github.com/raminetinati/aws-demo-textract-receipts/) developed illustrates how Data Scientists can use Textract to process, and structured data from documents, and then how to explore that data for further insights using Part of Speach Tagging (POS), Named Entity Recognition (NER), Word Co-occurence (manual), and Word Embeddings using Amazon SageMaker"s Built in Blazing Text Algorithm. 

By the end of this article, you should have learnt how to use Amazon Textract to ingest multiple images and obtain the results from the API, and how to enrich and analyse the results for processing purposes. We"re going to also be examing how to conduct simple Feature Engineering to reduce high cardinality data, and finally, how we can use Amazon SageMaker to build Word Embeddings (Vector Space Representations) of your data.

## Background

### Amazon Textract

>Amazon Textract is a service that automatically extracts text and data from scanned documents. Amazon Textract goes beyond simple optical character recognition (OCR) to also identify the contents of fields in forms and information stored in tables.

Many companies today extract data from documents and forms through manual data entry that’s slow and expensive or through simple optical character recognition (OCR) software that requires manual customization or configuration. Rules and workflows for each document and form often need to be hard-coded and updated with each change to the form or when dealing with multiple forms. If the form deviates from the rules, the output is often scrambled and unusable.

Amazon Textract overcomes these challenges by using machine learning to instantly “read” virtually any type of document to accurately extract text and data without the need for any manual effort or custom code. With Textract you can quickly automate document workflows, enabling you to process millions of document pages in hours. Once the information is captured, you can take action on it within your business applications to initiate next steps for a loan application or medical claims processing. Additionally, you can create smart search indexes, build automated approval workflows, and better maintain compliance with document archival rules by flagging data that may require redaction.

## Building a Solution: Receipt Processing

In order to illustrate the process of using Amazon Textract to build an OCR solution to build a real-world use case, we"re going to focus our efforts on developing a receipt processing solution which can extract data from receipts, independant on their structure and format. This is a common real-world problem which many organisations are facing, and without suitable automated processes, the overheads of manually reading and transcribing receipts can require a substantial number of human resources. 

One of the biggest challenges when building such solutions is being able to interpret the output of a OCR processed document, given that there labels are not available. Let"s take a second to consider why this is technically challenging. If we were to take a trivial example of a receipt processing solution which only had to interpret one type of receipt, from one merchant. For this use case, we could build a solution which has a dictonary of words specific to the products that the merchant sells, and a custom image processing pipeline which can detect the regions in the receipts which correspond to regions in the receipt with known content (e.g. the header, the main itemized section, the Sub-Total. the Total). However, now consider the scenario such as ours, where we are trying to build a solution which can process receipts from different Merchants, the receipts have many different structures and shapes, there is no consistency between content location (e.g. Total vs Header), the dictonary of words used will be extremely vast, and even the meaning of words, or language, may be different. Very quickly the technical scope expands, and no longer can you develop a rule based system, but you need to use data processing and mining techniques to make sence of the data.

For the use case in this article, we"re going to describe the process which a data science team would go through to develop the first stage in the pipeline of digitalizing and consuming the receipts information, which involves using Textract to perform Optical Character Recognition (OCR) on receipts. The process we"re going to cover in this post explores the necessary exploration and experimentation for extracting relevant information from unstructured, unlabelled data.

### Data Science Process

Without going off on a tangent and never returining, let"s think about the data science process that needs to be used when developing a solution such as this. Before jumping into concepts such as Machine Learning, Data Science teams will spent a vast majority of their time looking and manipulating the data sources in order to understand the shape and structure of the data they"re trying to proces (or potentially model in the future). This process is highly iterative, and requires both contextual knowledge of the domain (in this case, Merchant data), as as well data manipulation and transformation techniques to expose the underlying commonalities and patterns. As we will further in this post, this iterative process involves  adding additional analysis techniques as more information is discovered (For any social scientists out there, this is similar to snowball sampling), and then iterating over the initial analyses which were conducted in order to improve and refine the knwoledge about the dataset. As a result, the data science team will end up with many different experiments and findings as the data and methods of analysis improves. Each of these need to be tracked, documented, and then used to help support the underlying business requirement and needs.


### The Dataset

For this demonstration, we"re going to be using a dataset of 200 receipts, which has been made available by [ExpressExpense](https://expressexpense.com/blog/free-receipt-images-ocr-machine-learning-dataset/). The receipts data contains 200 different receipt images, with different backgrounds and types of receipts. This dataset is a great source of input for our solution, as we"re trying to build a processing pipeline with Amazon Textract that can support a wide spectrum of receipt types (e.g. they shouldn"t all come from the same Merchant). An example of the receipts can be seen below.

<RECEIPT IMAGE>

### Data Enrichment Tools

Data Enrichment is the processing of enhancing a source of data with additional data sources; this can be performed by custom processing of the data, or if available, by using 3rd party data (open or closed). The data enrichment process, can be applied to many different types of data, from images, video, audio, to text, and can be simple enrichments such as adding tags to a data point (e.g. Adding context to a word), or more complex enrichments, such as linking to external data sources, or perhaps to other sources within the data (e.g. constructing a graph of resources).

In our example, we"re going to be using two data enrichment processes in order to add additional knowledge about the content of our receipts, which inturn, will allow us to process the information more efficiently. 


### Part of Speech (POS) Tagging

Part of Speech tagging is the process of marking up a word in a corus as corresponding to a particular part of speech based on both its definition and its context. which is based on it"s relationship with other words in the sentance or phrase that it is located in. At its most primative, POS can be used to perform identification of words as nouns, verbs, adjectives, adverbs, etc.

POS can be extremely useful for text processing, especially when you are trying to find out the context or meaning of a sentance, phrase or paragraph. For more advances uses, POS allows the construction of Parse Trees, which allow for more complex processing and insights. For more information on this, take a look [here](https://en.wikipedia.org/wiki/Parse_tree)


### Named Entity Recognition (NER)

Named-entity recognition (NER) is a sub-task of information extraction that seeks to locate and classify named entity mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, and percentages.

There are multiple uses of NER across many different domains, from enriching news articles with tags for names, organizations, and locations, to improving our search results and search indexes. As the NER process relies on large corpuses of reference data, usually 3rd party libraries such as [SpaCy](https://spacy.io/) is used. 


##  Solutiom Walkthrough

In the following section we"re going to walkthrough the example solution which was build and can be found [here](https://github.com/raminetinati/aws-demo-textract-receipts/). The purpose of this walkthrough is not to give you a line-by-line review of the code (there"s documentation in the Notebook for this), but to demonstrate some of the high-level concepts which have been used for the data exploration and analysis, and how they support the extraction of information from unstructured data sources where the ground truth is not known.

### Step 1. Data Loading and Textract OCR Processing

The first step in our process will be to obtain our data, and then pass it to Amazon Textract to extract the text/words. In order to do this, we will first unzip the receipt images and then upload them to the S3 bucket, which was named in the config files. As we"re currently in experimental mode, the data source will come from a single pool of data (e.g. the tar file in the github repository), however, in the future, it may be that we"re pushing data to S3 from an external of source, which will then be used for further exploration.

For this example, we will use the ```detect_document_text``` [endpoint](https://docs.aws.amazon.com/textract/latest/dg/API_DetectDocumentText.html) available via the Textract API. Documentation can be found [here](https://textract.readthedocs.io/en/stable/). Prior to sending the data to the API, we first need to convert the images to their Byte array representation (a matrix of bytes), which will then be consumed by the Textract ```detect_document_text``` API endpoint.

After invoking the Textract endpoint, we are then returned a JSON response which contains the data(text) found within the image, along with additional metadata, such as the Confidene score of the Word or Line which has been identified, the geometry data which can be used to identify the bounding box, and the relationships with other items identified in the document.

Below is an example of two entries in the response, one showing the LINE entry (one more more words), and one for WORD (one word only). As shown in the relationships Array, the detected word *long* has an Id of *703dbf83-ec19-400d-a445-863271b2911c* which is found in the Relationships List of the LINE entry for text *Long Beach. CA 90804*. 


```json
{"BlockType": "LINE",
    "Confidence": 91.95721435546875,
    "Text": "Long Beach. CA 90804",
    "Geometry": {"BoundingBox": {"Width": 0.300077348947525,
      "Height": 0.039210282266139984,
      "Left": 0.38969704508781433,
      "Top": 0.3248765766620636},
     "Polygon": [{"X": 0.39162901043891907, "Y": 0.33458489179611206},
      {"X": 0.6894412636756897, "Y": 0.3248765766620636},
      {"X": 0.6897743940353394, "Y": 0.35538360476493835},
      {"X": 0.38969704508781433, "Y": 0.3640868663787842}]},
    "Id": "e991117e-c0a6-42c3-81ac-3778e41a65a7",
    "Relationships": [{"Type": "CHILD",
      "Ids": ["703dbf83-ec19-400d-a445-863271b2911c",
       "e8c98b04-7a1f-4363-9d6b-221a6d6039a0",
       "1d292d3b-4a8f-43f3-a70d-efde1d9fda85",
       "a938c8b2-cd63-4d3e-b48a-7b9a9e84554c"]}]},
...
{"BlockType": "WORD",
    "Confidence": 84.29407501220703,
    "Text": "Long",
    "Geometry": {"BoundingBox": {"Width": 0.060677558183670044,
      "Height": 0.024492284283041954,
      "Left": 0.3901515007019043,
      "Top": 0.33265504240989685},
     "Polygon": [{"X": 0.39162901043891907, "Y": 0.33458489179611206},
      {"X": 0.45082905888557434, "Y": 0.33265504240989685},
      {"X": 0.44968584179878235, "Y": 0.35537028312683105},
      {"X": 0.3901515007019043, "Y": 0.35714733600616455}]},
    "Id": "703dbf83-ec19-400d-a445-863271b2911c"},
}
```

Now that we have processed each of the images and have a digitalized version of our receipts, we"re now going to shift to processing and enriching this data with additional information to determine whether we can derive more context and meaning with our receipts.

### Step 2. Data Enrichment and Exploration

As discussed earlier in this article, the cleaning-enrichment-exploration is an iterative process which will involve several methods to understand the shape of our data at the micro (per document) and macro (collection of documents) level.

One of the first things that needs to be conducted is deciding whether to use the WORD or LINE elements fronm the Textract response. For this example we"re going to use the WORD elements in our response, as we don"t want to assume any relationship between our identified text prior to processing it. As there is the `Relationships` Element in our response, we can always reconstruct our data if we wish to do so.

#### Text Preprocessing and Stop Word Removals

If you"re familiar witih Natural Language Processing (NLP), then you might be familar with the pre-processing required to ensure the data is as clean as possible before using it for modelling or other purposes (e.g. Dashboard visualizations). NLP Pre-processing has many steps, including stemming and lemmatization (obtaining the root of the word, e.g. processes, process), as well as removing punctuation, language checking, stop word removalm, and tokenization (splitting sentences into words)

One of the most iterative processes in most NLP tasks is the development of the stop words. Stop words are common words in a corpus of text which are not useful for processing and analysis purposes, this they are removed. Common stop words are things like "is", "and", "the". The stripping of these common terms can be performed with libraries such as `nltk` (as we do in our example). However, in real-world use cases, usually domain specific lists are generated to remove words which are common. For our domain, merchant restaurant receipts, we have somewhat of a limited vocaluary which will be consistent across the receipts, these include terms such as "Total", "Server", "Tax", etc. Thus, we need to develop a list of stop words which help reduce the noise in our data. This is an iterative process and requieres the data scientists to iteratively process the words and examine the output to measure the effect of removing specific terms. It"s also important to note that there are approaches such as TF-IDF and LDA which can help reduce the need to remove commonly used terms across multiple documents, but in practice, domain stop words has benefits. 

In our example, we have generated a list of 100+ stop words which are commonly used in receipts, and for our tast, do not provide added analytical insights. These terms can be found in the `stopwords_custom.txt` document. This list, in combination with the `PortStemmer`, and `nltk`, for the first step in our text pre-processing pipe. Just to reinforce the iterative nature of the NLP process, the initial version of the `process_textract_responses_v1` method was only 19 lines long, 


```python
def process_textract_responses_v1(global_vars, data):
    stop_words = global_vars["stop_words"]
    records_enriched = {}
    for key,response in data.items():
        line_data = []
        confidences = []
        word = ""
        for item in response["Blocks"]:
            if item["BlockType"] == "WORD":                   
                #we need to normailise and remove punctuation.
            	word = item["Text"]
                if (word not in stop_words)==:
                    confidences.append({"word": word, "confidence": item["Confidence"]})
        #now we have the data
        record_data = process_textract_text(line_data, False)
        record_data["word_confidences"] = confidences
        record_data["response_raw"] = response
        records_enriched[key] = record_data
    return records_enriched
records_enriched = process_textract_responses(global_vars, textract_data)
```

However, if we examine the latest version of the method in the notebook, you"ll see that there are a lot more conditional statements and preprocessing steps used before the identified word is added to the processed list of words.

#### Data Enrichment with NLTK and SpaCy

As part of our processing pipeline, we"re using `NLTK` for performing Part-Of-Speech Tagging, and then `spaCy` for Named Entity Recognition. Whilst very simple to use (see below), POS and NER help add context and semantics to the data we"re trying to process. 

```python 
tokens = nltk.word_tokenize(words)
nltk_tagged_tokens = nltk.pos_tag(tokens)
```

Using the above code snipping on a string `This sentence has been processed by NLTK Part of Speech Tagger`, produces an output such as:

```json
[["This", "DT"], ["sentence", "NN"], ["has", "VBZ"], ["been", "VBN"], ["processed", "VBN"], ["by", "IN"], ["NLTK", "NNP"], ["Part", "NNP"], ["of", "IN"], ["Speech", "NNP"], ["Tagger", "NNP"]]
```
In just two lines of code we now have tags for each of the words identified in the image. Similarly, we can apply spaCy to extract recogized entities, with the above string, we would be returned the following:

```json
[["NLTK", "ORG"], ["Speech Tagger", "PRODUCT"]]
```

Based on these results, we can then start to bag our terms into different categories, and start to apply simple processing to the results, based on their tags. For instance, when we identify dates or cardinal values, we can then add some conditional statements, and based on some assumptions), select a value which could represent the total value of the bill, or the date of the transaction. Again, several assumptions are made at this step; take for instance the variable `max_value`, which is used to denote the maximum value found on the receipt. After a few stages of pre-processing and sanity checking, e.g. removing numbers such as barcodes, we take the maximum value as the bill value, however this can be problematic, as some receipts maximum value actually represent the cash which was given to pay the bill, e.g. 20 USD, where as the actual bill total was only 15.99 USD.


Acknowledging these limitations in our pre-processing and enrichment stages, we're able to proceed to now analyse and fuerther enhance our data in order to use it for other purposes.


#### Boundary Box Analysis

Visual inspection is a great tool to examine the output of Amazon Textract, and whilst you cannot do this at scale (e.g. all receipt images), being able to inspect the OCR results, especially when the confidence levels are low, is important. It's also good practice when performing data exploration and analysis to at some point in the process (earlier the better) to visually inspect the results of the processing, whether this be for text, images, or video.


#### Basic Analysis of Enriched Records

One of the most basic steps we can do to understand the results of our OCR process, is to examine both the data at the Micro (e.g. at the level of each record), and at the Macro (e.g. the dataset as a whole). For each the Micro and Macro, we need to apply different instruments of analysis, and both will provide different insights to how we can use our data.


- Macro Level Analysis: This typically involves looking at distributions of records, from the type of tokens we have, measures of skewness, or depending on the domain of the dataset, aspects such as timeseries or PMD plots. For our domain, we will use this exploratory step to understand the type of tokens we're commonly identifying within our dataset, as well as the distribution of confidence scores across our words. This is going to be an iterative process, as what we're aiming to do here is refine our custom dictionary of words that we dont want to include. Whilst it is hard to demonstrate how this iterative process happens, it's important to understand that the words in the ```stop_words_custom.txt``` file was not generated on a single pass, rather, it was an iterative process of viewing common words within the dataset, and based on the context of the data, we can keep or remove the words. A good example of this is the word `Server` or `Waiter`, for this domain (Receipt's data), this is going to be a common term, and for the purpose of our task, does not add value. That being said, if the task was to identify receipts where there was a `Server` or `Waiter` involved, then it would be benifical to keep this term. 

In the ```analyse_records``` method, a series of Macro level inspections on the data are conducted, such as examining the distribution of the confidence scores on the words across all the records, as well as the distribution of max values found, and spread of timestamps.

\Whilst these descriptive stats are quite rough and high-level, they provide some intuition on the processing pipeline we're building, and highlight any major flaws or errors in our steps. Using this visual inspection of the data, we can also refine the dictionary of stop words.

Take for example the output below, we can see that from the top 10 common words, there are perhaps futher processing steps required to ensure that we don't have duplicate terms like `0.00` and `$0.00`, or that we need a more refined approach to select the dates, given that there are only 200 records, and we have 211 dates in our table.

```sh
Total Non-unique Tokens 7755
Unique Tokens 4819
Top 10 Common Tokens ['0.00', 'chicken', '$0.00', '12.00', 'taco', 'coke', 'chees', 'grill', 'shrimp', 'dinner']
Highest Total Value 827.0 
Lowest Total Value 0.0 
Mean Total Value 67.59 
       max_values
count  200.000000
mean    67.592400
std    103.174735
min      0.000000
25%     18.825000
50%     38.545000
75%     73.265000
max    827.000000
Total Dates Found 211
```


















