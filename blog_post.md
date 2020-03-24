# Amazon Textract - Building a Receipt Processing Solution

## Overview

In [May 2019](https://press.aboutamazon.com/news-releases/news-release-details/aws-announces-general-availability-amazon-textract) Amazon announced the General Availability of Amnazon Textract, which is a fully managed service that uses machine learning to automatically extract text and data, including from tables and forms, in virtually any document – with no machine learning experience required. In this blog post, we're going to explore the use of Amazon Textract to build a Receipt Processing Solution, which uses images of different types of receipts, and demonstrates how to apply different methods to process the data.

For this solution, the [code](https://github.com/raminetinati/aws-demo-textract-receipts/) developed illustrates how Data Scientists can use Textract to process, and structured data from documents, and then how to explore that data for further insights using Part of Speach Tagging (POS), Named Entity Recognition (NER), Word Co-occurence (manual), and Word Embeddings using Amazon SageMaker's Built in Blazing Text Algorithm. 

By the end of this article, you should have learnt how to use Amazon Textract to ingest multiple images and obtain the results from the API, and how to enrich and analyse the results for processing purposes. We're going to also be examing how to conduct simple Feature Engineering to reduce high cardinality data, and finally, how we can use Amazon SageMaker to build Word Embeddings (Vector Space Representations) of your data.

## Background

### Amazon Textract

>Amazon Textract is a service that automatically extracts text and data from scanned documents. Amazon Textract goes beyond simple optical character recognition (OCR) to also identify the contents of fields in forms and information stored in tables.

Many companies today extract data from documents and forms through manual data entry that’s slow and expensive or through simple optical character recognition (OCR) software that requires manual customization or configuration. Rules and workflows for each document and form often need to be hard-coded and updated with each change to the form or when dealing with multiple forms. If the form deviates from the rules, the output is often scrambled and unusable.

Amazon Textract overcomes these challenges by using machine learning to instantly “read” virtually any type of document to accurately extract text and data without the need for any manual effort or custom code. With Textract you can quickly automate document workflows, enabling you to process millions of document pages in hours. Once the information is captured, you can take action on it within your business applications to initiate next steps for a loan application or medical claims processing. Additionally, you can create smart search indexes, build automated approval workflows, and better maintain compliance with document archival rules by flagging data that may require redaction.

## Building a Solution: Receipt Processing

In order to illustrate the process of using Amazon Textract to build an OCR solution to build a real-world use case, we're going to focus our efforts on developing a receipt processing solution which can extract data from receipts, independant on their structure and format. This is a common real-world problem which many organisations are facing, and without suitable automated processes, the overheads of manually reading and transcribing receipts can require a substantial number of human resources. 

One of the biggest challenges when building such solutions is being able to interpret the output of a OCR processed document, given that there labels are not available. Let's take a second to consider why this is technically challenging. If we were to take a trivial example of a receipt processing solution which only had to interpret one type of receipt, from one merchant. For this use case, we could build a solution which has a dictonary of words specific to the products that the merchant sells, and a custom image processing pipeline which can detect the regions in the receipts which correspond to regions in the receipt with known content (e.g. the header, the main itemized section, the Sub-Total. the Total). However, now consider the scenario such as ours, where we are trying to build a solution which can process receipts from different Merchants, the receipts have many different structures and shapes, there is no consistency between content location (e.g. Total vs Header), the dictonary of words used will be extremely vast, and even the meaning of words, or language, may be different. Very quickly the technical scope expands, and no longer can you develop a rule based system, but you need to use data processing and mining techniques to make sence of the data.

For the use case in this article, we're going to describe the process which a data science team would go through to develop the first stage in the pipeline of digitalizing and consuming the receipts information, which involves using Textract to perform Optical Character Recognition (OCR) on receipts. The process we're going to cover in this post explores the necessary exploration and experimentation for extracting relevant information from unstructured, unlabelled data.

### Data Science Process

Without going off on a tangent and never returining, let's think about the data science process that needs to be used when developing a solution such as this. Before jumping into concepts such as Machine Learning, Data Science teams will spent a vast majority of their time looking and manipulating the data sources in order to understand the shape and structure of the data they're trying to proces (or potentially model in the future). This process is highly iterative, and requires both contextual knowledge of the domain (in this case, Merchant data), as as well data manipulation and transformation techniques to expose the underlying commonalities and patterns. As we will further in this post, this iterative process involves  adding additional analysis techniques as more information is discovered (For any social scientists out there, this is similar to snowball sampling), and then iterating over the initial analyses which were conducted in order to improve and refine the knwoledge about the dataset. As a result, the data science team will end up with many different experiments and findings as the data and methods of analysis improves. Each of these need to be tracked, documented, and then used to help support the underlying business requirement and needs.


### The Dataset

For this demonstration, we're going to be using a dataset of 200 receipts, which has been made available by [ExpressExpense](https://expressexpense.com/blog/free-receipt-images-ocr-machine-learning-dataset/). The receipts data contains 200 different receipt images, with different backgrounds and types of receipts. This dataset is a great source of input for our solution, as we're trying to build a processing pipeline with Amazon Textract that can support a wide spectrum of receipt types (e.g. they shouldn't all come from the same Merchant). An example of the receipts can be seen below.

<RECEIPT IMAGE>

### Data Enrichment Tools

Data Enrichment is the processing of enhancing a source of data with additional data sources; this can be performed by custom processing of the data, or if available, by using 3rd party data (open or closed). The data enrichment process, can be applied to many different types of data, from images, video, audio, to text, and can be simple enrichments such as adding tags to a data point (e.g. Adding context to a word), or more complex enrichments, such as linking to external data sources, or perhaps to other sources within the data (e.g. constructing a graph of resources).

In our example, we're going to be using two data enrichment processes in order to add additional knowledge about the content of our receipts, which inturn, will allow us to process the information more efficiently. 


### Part of Speech (POS) Tagging

Part of Speech tagging is the process of marking up a word in a corus as corresponding to a particular part of speech based on both its definition and its context. which is based on it's relationship with other words in the sentance or phrase that it is located in. At its most primative, POS can be used to perform identification of words as nouns, verbs, adjectives, adverbs, etc.

POS can be extremely useful for text processing, especially when you are trying to find out the context or meaning of a sentance, phrase or paragraph. For more advances uses, POS allows the construction of Parse Trees, which allow for more complex processing and insights. For more information on this, take a look [here](https://en.wikipedia.org/wiki/Parse_tree)


### Named Entity Recognition (NER)

Named-entity recognition (NER) is a sub-task of information extraction that seeks to locate and classify named entity mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, and percentages.

There are multiple uses of NER across many different domains, from enriching news articles with tags for names, organizations, and locations, to improving our search results and search indexes. As the NER process relies on large corpuses of reference data, usually 3rd party libraries such as [SpaCy](https://spacy.io/) is used. 




## Example Walkthrough


















