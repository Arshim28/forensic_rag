import os
import argparse
import json
import pandas as pd
from typing import List, Dict, Any
from ocr_vector_store import OCRVectorStore
from google import genai
from google.genai import types
from config import GOOGLE_API_KEY
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

client = genai.Client()


def process_document(pdf_path: str, vector_store_dir: str, reprocess: bool = False) -> OCRVectorStore:
    vector_store = OCRVectorStore(
        index_type="HNSW",
        chunk_size=8000,
        chunk_overlap=400
    )
    
    if reprocess or not os.path.exists(vector_store_dir) or not os.listdir(vector_store_dir):
        print(f"Processing document: {pdf_path}")
        vector_store.add_document(pdf_path)
        os.makedirs(vector_store_dir, exist_ok=True)
        vector_store.save(vector_store_dir)
        print(f"Vector store saved to {vector_store_dir}")
    else:
        print(f"Loading existing vector store from {vector_store_dir}")
        vector_store.load(vector_store_dir)
    
    return vector_store

def extract_key_topics(mosaic_store: OCRVectorStore) -> List[str]:
    """Extract key topics to investigate from the SFC Mosaic Note"""
    print("Extracting key topics from SFC Mosaic Note")
    
    topic_query = """
    Analyze this document thoroughly and identify the 10 most important topics or areas that should be
    verified for consistency with other documents about the same company. Focus on factual information
    that might differ between documents like company details, financial figures, management information,
    business model, etc. Format your response as a simple list of topics, one per line.
    """
    
    results = mosaic_store.answer_question(topic_query, k=20)
    
    # Extract topics using LLM
    context = ""
    for result in results:
        context += f"{result['text'][:500]}...\n\n"
    
    extract_prompt = f"""
    Based on the following excerpts from the SFC Mosaic Note, identify the 10 most important topics
    or areas that should be verified for consistency with other documents (DRHP and Credit Rating Report).
    
    {context}
    
    Focus on factual information that might differ between documents like:
    - Company details (name, incorporation, structure)
    - Financial figures (revenue, profit, debt)
    - Management information (key personnel, board)
    - Business model (products, services, markets)
    - Risks and challenges
    
    Format your response as a simple numbered list of topics, one per line.
    Example:
    1. Company incorporation date and legal name
    2. Revenue figures for fiscal years 2022-2024
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=extract_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0.2
            )
        )
        
        if response and hasattr(response, 'text'):
            topics = []
            for line in response.text.strip().split('\n'):
                if line.strip():
                    # Remove any numbering and strip whitespace
                    topic = line.strip()
                    if '.' in topic[:4]:  # Remove numbering if present
                        topic = topic.split('.', 1)[1].strip()
                    topics.append(topic)
            
            print(f"Extracted {len(topics)} key topics")
            return topics
        else:
            print("Warning: Failed to extract topics")
            return default_topics()
    except Exception as e:
        print(f"Error extracting topics: {str(e)}")
        return default_topics()

def default_topics() -> List[str]:
    """Fallback list of topics to investigate"""
    return [
        "Company legal name and incorporation details",
        "Revenue figures for recent fiscal years",
        "Profit margins and financial ratios",
        "Management team and board composition",
        "Business model and revenue streams",
        "Major projects and contracts",
        "Debt obligations and financial liabilities",
        "Risk factors and challenges",
        "Shareholding pattern and ownership structure",
        "Regulatory compliance and approvals"
    ]

def generate_specific_queries(topics: List[str]) -> List[str]:
    """Generate specific queries based on the extracted topics"""
    print("Generating specific queries based on extracted topics")
    
    all_queries = []
    
    for topic in topics:
        query_prompt = f"""
        Generate 3-5 specific, detailed queries to identify factual inconsistencies when comparing 
        documents about a company. These queries should be based on the topic: "{topic}"
        
        Focus on queries that would identify factual inconsistencies in:
        - Exact figures, dates, names, and percentages
        - Specific claims about business operations
        - Regulatory status and compliance
        - Financial performance metrics
        
        Format each query as a complete, specific question that can be directly answered from the documents.
        
        Examples of good queries:
        - "What was the exact revenue figure for fiscal year ending March 31, 2023?"
        - "Who is listed as the CEO and what is their full name and background?"
        - "What is the company's claimed market share percentage in the wastewater treatment market?"
        
        Return only the list of questions, one per line, without any explanations or numbering.
        """
        
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=query_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1000,
                    temperature=0.2
                )
            )
            
            if response and hasattr(response, 'text'):
                topic_queries = []
                for line in response.text.strip().split('\n'):
                    if line.strip():
                        query = line.strip()
                        # Remove any numbers or bullets
                        if query[0].isdigit() and '. ' in query[:4]:
                            query = query.split('. ', 1)[1]
                        elif query.startswith('- '):
                            query = query[2:]
                        topic_queries.append(query)
                
                print(f"  Generated {len(topic_queries)} queries for topic: {topic}")
                all_queries.extend(topic_queries)
            else:
                print(f"  Warning: Failed to generate queries for topic: {topic}")
        except Exception as e:
            print(f"  Error generating queries for topic {topic}: {str(e)}")
    
    # Remove duplicates
    all_queries = list(set(all_queries))
    print(f"Generated {len(all_queries)} total unique queries")
    
    # Limit to a manageable number
    if len(all_queries) > 40:
        print(f"Limiting to 40 most relevant queries")
        all_queries = all_queries[:40]
    
    return all_queries

def query_all_documents(queries: List[str], doc_stores: Dict[str, OCRVectorStore]) -> Dict[str, Dict[str, List]]:
    """Run all queries against all documents and store the results"""
    print("Running queries against all documents")
    
    results = {}
    
    for i, query in enumerate(queries):
        print(f"  Query {i+1}/{len(queries)}: {query}")
        query_results = {}
        
        for name, store in doc_stores.items():
            query_results[name] = store.answer_question(query, k=5)
        
        results[query] = query_results
    
    return results

def analyze_for_inconsistencies(query_results: Dict[str, Dict[str, List]]) -> List[Dict]:
    """Analyze the query results for inconsistencies between documents"""
    print("Analyzing results for inconsistencies")
    
    inconsistencies = []
    
    for query, doc_results in query_results.items():
        print(f"  Analyzing query: {query}")
        
        doc_names = list(doc_results.keys())
        if len(doc_names) < 2:
            print("    Skipping: Not enough documents to compare")
            continue
        
        # Prepare summaries for each document
        doc_summaries = {}
        for doc_name, results in doc_results.items():
            if not results:
                doc_summaries[doc_name] = "No relevant information found."
                continue
            
            # Get a summary of this document's answer
            summary_prompt = f"""
            Summarize the following excerpts from {doc_name} that respond to the query: "{query}"
            Keep your summary to 1-2 sentences focusing on the specific answer to the query.
            Focus only on facts, figures, dates, names, and other objective information.
            
            Excerpts:
            {[f'- {r["text"][:300]}...' for r in results[:3]]}
            """
            
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=summary_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=500,
                        temperature=0.1
                    )
                )
                
                if response and hasattr(response, 'text'):
                    doc_summaries[doc_name] = response.text.strip()
                else:
                    doc_summaries[doc_name] = "Failed to generate summary."
            except Exception as e:
                print(f"    Error generating summary for {doc_name}: {str(e)}")
                doc_summaries[doc_name] = "Error generating summary."
        
        # Check for inconsistencies between document summaries
        comparison_prompt = f"""
        Analyze these document excerpts for inconsistencies. They all respond to the query: "{query}"
        
        {", ".join([f"{name}: {summary}" for name, summary in doc_summaries.items()])}
        
        Is there any inconsistency, contradiction, or significant difference between these documents?
        
        Answer in this exact format:
        INCONSISTENCY: [Yes/No]
        DETAILS: [If Yes, describe the specific inconsistency in 1-2 sentences]
        SEVERITY: [If Yes, rate as "RED" (major factual contradiction), "ORANGE" (moderate difference), or "YELLOW" (minor variation)]
        """
        
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=comparison_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=500,
                    temperature=0.1
                )
            )
            
            if response and hasattr(response, 'text'):
                analysis = response.text.strip()
                
                # Parse the analysis
                is_inconsistent = "INCONSISTENCY: Yes" in analysis
                
                if is_inconsistent:
                    # Extract details
                    details = ""
                    details_match = analysis.split("DETAILS:", 1)
                    if len(details_match) > 1:
                        details = details_match[1].split("SEVERITY:", 1)[0].strip()
                    
                    # Extract severity
                    severity = "ORANGE"  # Default
                    severity_match = analysis.split("SEVERITY:", 1)
                    if len(severity_match) > 1:
                        severity_text = severity_match[1].strip()
                        if "RED" in severity_text:
                            severity = "RED"
                        elif "YELLOW" in severity_text:
                            severity = "YELLOW"
                    
                    inconsistencies.append({
                        "query": query,
                        "responses": doc_summaries,
                        "details": details,
                        "severity": severity
                    })
                    
                    print(f"    ⚠️ Inconsistency found: {severity}")
                else:
                    print(f"    ✓ No inconsistency found")
            else:
                print(f"    ⚠️ Warning: Failed to analyze")
        except Exception as e:
            print(f"    ⚠️ Error analyzing: {str(e)}")
    
    print(f"Found {len(inconsistencies)} potential inconsistencies")
    return inconsistencies

def generate_report(inconsistencies: List[Dict], output_file: str):
    """Generate a comprehensive report of the findings"""
    print(f"Generating report to {output_file}")
    
    # Group inconsistencies by severity
    by_severity = {"RED": [], "ORANGE": [], "YELLOW": []}
    for inc in inconsistencies:
        severity = inc.get("severity", "ORANGE")
        by_severity[severity].append(inc)
    
    # Count statistics
    total = len(inconsistencies)
    red_count = len(by_severity["RED"])
    orange_count = len(by_severity["ORANGE"])
    yellow_count = len(by_severity["YELLOW"])
    
    # Generate executive summary
    summary_prompt = f"""
    Generate an executive summary for a document inconsistency analysis report with these findings:
    - Total inconsistencies: {total}
    - RED (major) inconsistencies: {red_count}
    - ORANGE (moderate) inconsistencies: {orange_count}
    - YELLOW (minor) inconsistencies: {yellow_count}
    
    Examples of inconsistencies:
    {[inc["query"] + ": " + inc["details"] for inc in inconsistencies[:3]]}
    
    The executive summary should:
    1. Explain the purpose of the analysis
    2. Summarize the key findings
    3. Provide a high-level assessment of the reliability of information
    4. Offer recommendations for how to interpret these inconsistencies
    
    Keep the summary to 400-500 words and maintain a professional, balanced tone.
    """
    
    try:
        exec_summary = "## Executive Summary\n\n"
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=summary_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=2000,
                temperature=0.2
            )
        )
        
        if response and hasattr(response, 'text'):
            exec_summary += response.text.strip()
        else:
            exec_summary += "Failed to generate executive summary."
    except Exception as e:
        print(f"Error generating executive summary: {str(e)}")
        exec_summary += f"Error generating executive summary: {str(e)}"
    
    # Create markdown report
    md_report = "# Document Inconsistency Analysis Report\n\n"
    md_report += exec_summary
    md_report += "\n\n## Analysis Summary\n\n"
    md_report += f"- **Total Inconsistencies**: {total}\n"
    md_report += f"- **RED (Major)**: {red_count}\n"
    md_report += f"- **ORANGE (Moderate)**: {orange_count}\n"
    md_report += f"- **YELLOW (Minor)**: {yellow_count}\n\n"
    
    # Add RED inconsistencies
    if red_count > 0:
        md_report += "## RED (Major) Inconsistencies\n\n"
        for i, inc in enumerate(by_severity["RED"]):
            md_report += f"### R{i+1}: {inc['query']}\n\n"
            md_report += f"**Details**: {inc['details']}\n\n"
            md_report += "**Document Responses**:\n"
            for doc, response in inc['responses'].items():
                md_report += f"- **{doc}**: {response}\n"
            md_report += "\n---\n\n"
    
    # Add ORANGE inconsistencies
    if orange_count > 0:
        md_report += "## ORANGE (Moderate) Inconsistencies\n\n"
        for i, inc in enumerate(by_severity["ORANGE"]):
            md_report += f"### O{i+1}: {inc['query']}\n\n"
            md_report += f"**Details**: {inc['details']}\n\n"
            md_report += "**Document Responses**:\n"
            for doc, response in inc['responses'].items():
                md_report += f"- **{doc}**: {response}\n"
            md_report += "\n---\n\n"
    
    # Add YELLOW inconsistencies
    if yellow_count > 0:
        md_report += "## YELLOW (Minor) Inconsistencies\n\n"
        for i, inc in enumerate(by_severity["YELLOW"]):
            md_report += f"### Y{i+1}: {inc['query']}\n\n"
            md_report += f"**Details**: {inc['details']}\n\n"
            md_report += "**Document Responses**:\n"
            for doc, response in inc['responses'].items():
                md_report += f"- **{doc}**: {response}\n"
            md_report += "\n---\n\n"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(md_report)
    
    # Also create CSV for easier analysis
    csv_file = output_file.replace('.md', '.csv')
    
    # Create a dataframe from inconsistencies
    rows = []
    for inc in inconsistencies:
        row = {
            'Query': inc['query'],
            'Severity': inc['severity'],
            'Details': inc['details']
        }
        
        # Add document responses
        for doc, response in inc['responses'].items():
            row[f'{doc} Response'] = response
        
        rows.append(row)
    
    # Create DataFrame and save to CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        print(f"CSV report saved to {csv_file}")
    
    print(f"Report saved to {output_file}")
    
    return {
        'total': total,
        'red': red_count,
        'orange': orange_count,
        'yellow': yellow_count,
        'report_file': output_file,
        'csv_file': csv_file
    }

def main():
    parser = argparse.ArgumentParser(description="Simple Document Inconsistency Analyzer")
    parser.add_argument("--mosaic", required=True, help="Path to the SFC Mosaic Note PDF")
    parser.add_argument("--drhp", required=True, help="Path to the DRHP PDF")
    parser.add_argument("--rating", required=True, help="Path to the Credit Rating Report PDF")
    parser.add_argument("--output", default="simple_inconsistency_report.md", help="Output file for the report")
    parser.add_argument("--reprocess", action="store_true", help="Force reprocessing PDFs")
    
    args = parser.parse_args()
    
    # Process all documents
    print("\n=== PROCESSING DOCUMENTS ===\n")
    mosaic_store = process_document(args.mosaic, "vector_store_mosaic", args.reprocess)
    drhp_store = process_document(args.drhp, "vector_store_drhp", args.reprocess)
    rating_store = process_document(args.rating, "vector_store_rating", args.reprocess)
    
    # Set up document stores
    doc_stores = {
        "SFC Mosaic Note": mosaic_store,
        "DRHP": drhp_store,
        "Credit Rating Report": rating_store
    }
    
    # Extract key topics from the primary document
    print("\n=== EXTRACTING KEY TOPICS ===\n")
    topics = extract_key_topics(mosaic_store)
    
    # Generate specific queries based on topics
    print("\n=== GENERATING SPECIFIC QUERIES ===\n")
    queries = generate_specific_queries(topics)
    
    # Query all documents
    print("\n=== QUERYING ALL DOCUMENTS ===\n")
    query_results = query_all_documents(queries, doc_stores)
    
    # Analyze for inconsistencies
    print("\n=== ANALYZING FOR INCONSISTENCIES ===\n")
    inconsistencies = analyze_for_inconsistencies(query_results)
    
    # Generate report
    print("\n=== GENERATING REPORT ===\n")
    report_stats = generate_report(inconsistencies, args.output)
    
    print("\n=== ANALYSIS COMPLETE ===\n")
    print(f"Found {report_stats['total']} total inconsistencies:")
    print(f"- RED (Major): {report_stats['red']}")
    print(f"- ORANGE (Moderate): {report_stats['orange']}")
    print(f"- YELLOW (Minor): {report_stats['yellow']}")
    
    print(f"\nReport saved to {report_stats['report_file']}")
    print(f"CSV report saved to {report_stats['csv_file']}")

if __name__ == "__main__":
    main()