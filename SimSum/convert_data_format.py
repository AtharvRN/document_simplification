import csv

# Input file containing the text data
# input_file = "data/wiki_doc/wiki_doc.test.complex"
complex_file = "data/D_wiki/D_wiki.test.complex"
simple_file = "data/D_wiki/D_wiki.test.complex"

# complex_file = "data/plaba/plaba.test.complex"
# simple_file = "data/plaba/plaba.test.simple"
# output_csv_sentences = "data_pg/wiki_doc/wikidocs_sents_test.csv"
# output_csv_documents = "data_pg/wiki_doc/wikidocs_docs_test.csv"

output_csv_sentences = "data_pg/D_wiki/DWiki_sents_test.csv"
output_csv_documents = "data_pg/D_wiki/Dwiki_docs_test.csv"

# output_csv_sentences = "data_pg/plaba/plaba_sents_test.csv"
# output_csv_documents = "data_pg/plaba/plaba_docs_test.csv"

# Open input files for complex and simple sentences
with open(complex_file, "r", encoding="utf-8") as complex_infile, \
     open(simple_file, "r", encoding="utf-8") as simple_infile, \
     open(output_csv_sentences, "w", newline="", encoding="utf-8") as sentence_outfile, \
     open(output_csv_documents, "w", newline="", encoding="utf-8") as document_outfile:
    
    # Writer for combined sentence CSV
    sentence_writer = csv.writer(sentence_outfile)
    sentence_writer.writerow(["pair_id", "sent_id", "complex", "simple", "doc_pos", "doc_len"])  # Write header row for sentences
    
    # Writer for combined document CSV
    document_writer = csv.writer(document_outfile)
    document_writer.writerow(["pair_id", "complex", "simple"])  # Write header row for documents
    
    pair_id = 1  # Initialize pair_id
    
    # Iterate through each line in the complex and simple files simultaneously
    for complex_line, simple_line in zip(complex_infile, simple_infile):
        complex_sentences = complex_line.strip().split(". ")  # Split complex line into sentences
        simple_sentences = simple_line.strip().split(". ")  # Split simple line into sentences
        total_sents = len(complex_sentences)  # Total number of sentences in the document
        
        # Write each complex and corresponding simple sentence to the sentence CSV
        for sent_id, (complex_sentence, simple_sentence) in enumerate(zip(complex_sentences, simple_sentences), start=0):
            doc_pos = sent_id / total_sents  # Position of the sentence in the document
            complex_sentence = complex_sentence[:-1] + "." if complex_sentence.endswith(".") else complex_sentence  # Ensure the sentence ends with a full stop
            simple_sentence = simple_sentence[:-1] + "." if simple_sentence.endswith(".") else simple_sentence  # Ensure the sentence ends with a full stop
            sentence_writer.writerow([pair_id, sent_id, complex_sentence, simple_sentence, doc_pos, total_sents])
        
        # Add "<s>" after each full stop and write document-level data to the document CSV
        complex_document_text = " <s> ".join(complex_sentences) + " <s>"  # Join complex sentences and add "<s>" after each full stop
        simple_document_text = " ".join(simple_sentences)   # Join simple sentences and add "<s>" after each full stop
        document_writer.writerow([pair_id, complex_document_text, simple_document_text])
        
        pair_id += 1  # Increment pair_id for each document

print("CSV files with combined complex and simple sentences and documents have been generated successfully.")