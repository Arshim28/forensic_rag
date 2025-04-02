import os
import argparse
import pandas as pd
import re
from typing import List, Dict, Any
from ocr_vector_store import OCRVectorStore
from google import genai
from google.genai import types
from config import GOOGLE_API_KEY
from dotenv import load_dotenv
from md_to_pdf import convert_markdown_to_pdf

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

def extract_key_topics(mosaic_store: OCRVectorStore, max_retries: int = 3) -> List[str]:
    print("Extracting key topics from SFC Mosaic Note")
    
    topic_query = """
    Analyze this document thoroughly and identify the 10 most important topics or areas that should be
    verified for consistency with other documents about the same company. Focus on factual information
    that might differ between documents like company details, financial figures, management information,
    business model, etc. Format your response as a simple list of topics, one per line.
    """
    
    results = mosaic_store.answer_question(topic_query, k=50)
    
    context = ""
    for result in results:
        context += f"{result['text'][:1000]}...\n\n"
    
    extract_prompt = f"""
    Based on the following excerpts from the SFC Mosaic Note, identify the 10 most important topics
    or areas that should be verified for consistency with other documents (DRHP and Credit Rating Report).
    
    {context}
    
    Focus on factual information that might differ between documents like:
    - Company details (name, incorporation, structure)
    - Financial figures (revenue, profit, debt). If you are unsure about the units or any other related aspect, flag as yellow only!
    - Management information (key personnel, board). Do not create confusion between spelling mistakes in people's name
    - Business model (products, services, markets)
    - Risks and challenges
    
    Format your response as a simple numbered list of topics, one per line.
    """
    
    # Try with gemini-2.5-pro-exp first, with retries
    for retry in range(max_retries):
        try:
            print(f"Attempt {retry+1}/{max_retries} with gemini-2.5-pro-exp-03-25")
            response = client.models.generate_content(
                model="gemini-2.5-pro-exp-03-25",
                contents=extract_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1000,
                    temperature=0.1
                )
            )
            
            if response and hasattr(response, 'text') and response.text:
                topics = []
                for line in response.text.strip().split('\n'):
                    if line.strip():
                        topic = line.strip()
                        if '.' in topic[:4]:  
                            topic = topic.split('.', 1)[1].strip()
                        topics.append(topic)
                
                if topics:
                    print(f"Extracted {len(topics)} key topics")
                    return topics
                
            print(f"Warning: Empty response on retry {retry+1}/{max_retries}. Retrying...")
            
        except Exception as e:
            print(f"Error on retry {retry+1}/{max_retries}: {str(e)}. Retrying...")
    
    # Fallback to gemini-1.5-pro if gemini-2.5-pro-exp fails
    try:
        print("Falling back to gemini-1.5-pro model...")
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=extract_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            topics = []
            for line in response.text.strip().split('\n'):
                if line.strip():
                    topic = line.strip()
                    if '.' in topic[:4]:  
                        topic = topic.split('.', 1)[1].strip()
                    topics.append(topic)
            
            if topics:
                print(f"Extracted {len(topics)} key topics using fallback model")
                return topics
    except Exception as e:
        print(f"Error with fallback model: {str(e)}")
    
    # Final fallback to gemini-2.0-flash
    try:
        print("Falling back to gemini-2.0-flash model...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=extract_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            topics = []
            for line in response.text.strip().split('\n'):
                if line.strip():
                    topic = line.strip()
                    if '.' in topic[:4]:  
                        topic = topic.split('.', 1)[1].strip()
                    topics.append(topic)
            
            if topics:
                print(f"Extracted {len(topics)} key topics using gemini-2.0-flash")
                return topics
    except Exception as e:
        print(f"Error with gemini-2.0-flash model: {str(e)}")
    
    print("Using default topics as all extraction attempts failed")
    return default_topics()

def default_topics() -> List[str]:
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

def generate_specific_queries(topics: List[str], max_retries: int = 2) -> List[str]:
    print("Generating specific queries based on extracted topics")
    
    all_queries = []
    
    for topic in topics:
        query_prompt = f"""
        Generate 3-5 specific, detailed queries to identify factual inconsistencies when comparing 
        documents about a company. These queries should be based on the topic: "{topic}"
        
        Focus on queries that would identify:
        - Information mentioned in one document but missing in another
        - Direct contradictions in factual information
        - Exact figures, dates, names, and percentages that may differ
        - Specific claims about business operations
        
        Format each query as a complete, specific question that can be directly answered from the documents.
        
        Examples of good queries:
        - "What was the exact revenue figure for fiscal year ending March 31, 2023?"
        - "Who is listed as the CEO and what is their full name and background?"
        - "What is the company's claimed market share percentage in the wastewater treatment market?"
        
        Return only the list of questions, one per line, without any explanations or numbering.
        """
        
        # Try with gemini-1.5-pro first, with retries
        success = False
        for retry in range(max_retries):
            try:
                print(f"  Attempt {retry+1}/{max_retries} with gemini-1.5-pro for topic: {topic}")
                response = client.models.generate_content(
                    model="gemini-1.5-pro",
                    contents=query_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=1000,
                        temperature=0.2
                    )
                )
                
                if response and hasattr(response, 'text') and response.text:
                    topic_queries = []
                    for line in response.text.strip().split('\n'):
                        if line.strip():
                            query = line.strip()
                            if query[0].isdigit() and '. ' in query[:4]:
                                query = query.split('. ', 1)[1]
                            elif query.startswith('- '):
                                query = query[2:]
                            topic_queries.append(query)
                    
                    if topic_queries:
                        print(f"  Generated {len(topic_queries)} queries for topic: {topic}")
                        all_queries.extend(topic_queries)
                        success = True
                        break
                    else:
                        print(f"  Warning: Empty response on retry {retry+1}/{max_retries}. Retrying...")
                else:
                    print(f"  Warning: Failed response on retry {retry+1}/{max_retries}. Retrying...")
            except Exception as e:
                print(f"  Error on retry {retry+1}/{max_retries} for topic {topic}: {str(e)}")
        
        # Fallback to gemini-2.0-flash if gemini-1.5-pro fails
        if not success:
            try:
                print(f"  Falling back to gemini-2.0-flash for topic: {topic}")
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=query_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=1000,
                        temperature=0.2
                    )
                )
                
                if response and hasattr(response, 'text') and response.text:
                    topic_queries = []
                    for line in response.text.strip().split('\n'):
                        if line.strip():
                            query = line.strip()
                            if query[0].isdigit() and '. ' in query[:4]:
                                query = query.split('. ', 1)[1]
                            elif query.startswith('- '):
                                query = query[2:]
                            topic_queries.append(query)
                    
                    if topic_queries:
                        print(f"  Generated {len(topic_queries)} queries using fallback model for topic: {topic}")
                        all_queries.extend(topic_queries)
                    else:
                        print(f"  Warning: Failed to generate queries for topic: {topic} using fallback model")
                else:
                    print(f"  Warning: Failed to generate queries for topic: {topic} using fallback model")
            except Exception as e:
                print(f"  Error generating queries for topic {topic} using fallback model: {str(e)}")
    
    all_queries = list(set(all_queries))
    print(f"Generated {len(all_queries)} total unique queries")
    
    if len(all_queries) > 40:
        print(f"Limiting to 40 most relevant queries")
        all_queries = all_queries[:40]
    
    return all_queries

def query_documents(queries: List[str], primary_store: OCRVectorStore, 
                   secondary_store: OCRVectorStore, comparison_name: str) -> Dict[str, Dict[str, List]]:
    primary_name = "SFC Mosaic Note"
    secondary_name = comparison_name
    
    print(f"Running queries against {primary_name} and {secondary_name}")
    
    results = {}
    
    for i, query in enumerate(queries):
        print(f"  Query {i+1}/{len(queries)}: {query}")
        query_results = {}
        
        query_results[primary_name] = primary_store.answer_question(query, k=30)
        query_results[secondary_name] = secondary_store.answer_question(query, k=30)
        
        results[query] = query_results
    
    return results

def analyze_document_differences(query_results: Dict[str, Dict[str, List]], comparison_name: str, max_retries: int = 2) -> List[Dict]:
    print(f"Analyzing differences between SFC Mosaic Note and {comparison_name}")
    
    primary_name = "SFC Mosaic Note"
    secondary_name = comparison_name
    
    differences = []
    
    for query, doc_results in query_results.items():
        print(f"  Analyzing query: {query}")
        
        doc_summaries = {}
        doc_metadata = {}
        doc_text_samples = {}
        
        for doc_name, results in doc_results.items():
            if not results:
                doc_summaries[doc_name] = "No relevant information found."
                continue
            
            if results and len(results) > 0:
                doc_metadata[doc_name] = results[0].get('metadata', {})
            
            doc_text_samples[doc_name] = [r["text"][:300] for r in results[:3]]
            
            summary_prompt = f"""
            Summarize the following excerpts from {doc_name} that respond to the query: "{query}"
            
            Keep your summary to 2 sentences focusing on the specific answer to the query.
            Focus only on facts, figures, dates, names, and other objective information.
            Always specify currency units (₹, $, etc.) for financial figures.
            Always include time periods for any financial or operational data.
            
            Excerpts:
            {[f'- {r["text"][:1000]}...' for r in results[:3]]}
            """
            
            # Try with gemini-2.0-flash first
            summary_success = False
            for retry in range(max_retries):
                try:
                    print(f"    Generating summary for {doc_name}, attempt {retry+1}/{max_retries}")
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=summary_prompt,
                        config=types.GenerateContentConfig(
                            max_output_tokens=1500,
                            temperature=0.1
                        )
                    )
                    
                    if response and hasattr(response, 'text') and response.text:
                        doc_summaries[doc_name] = response.text.strip()
                        summary_success = True
                        break
                    else:
                        print(f"    Empty response on retry {retry+1}/{max_retries}")
                except Exception as e:
                    print(f"    Error generating summary on retry {retry+1}/{max_retries}: {str(e)}")
            
            # Fall back to a simpler approach if all retries fail
            if not summary_success:
                print(f"    All summary generation attempts failed, using fallback for {doc_name}")
                try:
                    # Simple fallback: just concatenate first 300 chars from each result
                    simple_summary = " ".join([r["text"][:300] + "..." for r in results[:2]])
                    doc_summaries[doc_name] = simple_summary[:500] + "..."
                except Exception as e:
                    print(f"    Even fallback summary failed: {str(e)}")
                    doc_summaries[doc_name] = "Error generating summary."
        
        # Skip if both documents have no relevant information
        if ("No relevant information found" in doc_summaries.get(primary_name, "") and 
            "No relevant information found" in doc_summaries.get(secondary_name, "")):
            print(f"    ✓ No information found in either document")
            continue
            
        comparison_prompt = f"""
        Analyze these document excerpts for discrepancies based on the query: "{query}"
        
        SFC Mosaic Note: {doc_summaries.get(primary_name, "No data available")}
        {secondary_name}: {doc_summaries.get(secondary_name, "No data available")}
        
        ONLY flag a discrepancy if:
        1. The Mosaic report mentions something that is NOWHERE mentioned in the {secondary_name}
        2. The Mosaic report MISSES something that is clearly mentioned in the {secondary_name}
        
        IGNORE cases where both documents don't mention relevant information.
        
        IS THERE A DISCREPANCY BASED ON THESE SPECIFIC CRITERIA? Answer Yes or No first.
        If Yes, provide:
        
        TYPE: [Either "Mosaic includes information not in {secondary_name}" OR "Mosaic misses information from {secondary_name}"]
        DETAILS: [Describe the specific discrepancy in exactly 1 sentence, being very precise]
        SEVERITY: [Rate as "RED" (major factual contradiction that impacts investment decisions), "ORANGE" (moderate difference needing clarification), or "YELLOW" (minor variation with limited impact)]
        
        If you are doubtful or uncertain, rate the severity as YELLOW and note your uncertainty.
        """
        
        # Try with gemini-2.5-pro-exp first, then fall back if needed
        analysis_success = False
        for retry in range(max_retries):
            try:
                print(f"    Analyzing discrepancies, attempt {retry+1}/{max_retries}")
                response = client.models.generate_content(
                    model="gemini-2.5-pro-exp-03-25",
                    contents=comparison_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=1500,
                        temperature=0.3
                    )
                )
                
                if response and hasattr(response, 'text') and response.text:
                    analysis = response.text.strip()
                    
                    is_discrepancy = "Yes" in analysis.split("\n")[0]
                    
                    if is_discrepancy:
                        discrepancy_type = ""
                        type_match = analysis.split("TYPE:", 1)
                        if len(type_match) > 1:
                            discrepancy_type = type_match[1].split("\n", 1)[0].strip()
                        
                        details = ""
                        details_match = analysis.split("DETAILS:", 1)
                        if len(details_match) > 1:
                            details = details_match[1].split("SEVERITY:", 1)[0].strip()
                        
                        severity = "YELLOW"  # Default to YELLOW for uncertain cases
                        severity_match = analysis.split("SEVERITY:", 1)
                        if len(severity_match) > 1:
                            severity_text = severity_match[1].strip()
                            if "RED" in severity_text:
                                severity = "RED"
                            elif "ORANGE" in severity_text:
                                severity = "ORANGE"
                        
                        differences.append({
                            "query": query,
                            "responses": doc_summaries,
                            "type": discrepancy_type,
                            "details": details,
                            "severity": severity,
                            "metadata": doc_metadata,
                            "text_samples": doc_text_samples,
                            "comparison_type": comparison_name
                        })
                        
                        print(f"    ⚠️ Discrepancy found: {severity}")
                    else:
                        print(f"    ✓ No discrepancy found based on criteria")
                    
                    analysis_success = True
                    break
                else:
                    print(f"    Empty analysis response on retry {retry+1}/{max_retries}")
            except Exception as e:
                print(f"    Error analyzing on retry {retry+1}/{max_retries}: {str(e)}")
        
        # Fall back to gemini-1.5-pro if all retries fail
        if not analysis_success:
            try:
                print(f"    Falling back to gemini-1.5-pro for analysis")
                response = client.models.generate_content(
                    model="gemini-1.5-pro",
                    contents=comparison_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=1500,
                        temperature=0.3
                    )
                )
                
                if response and hasattr(response, 'text') and response.text:
                    analysis = response.text.strip()
                    
                    is_discrepancy = "Yes" in analysis.split("\n")[0]
                    
                    if is_discrepancy:
                        discrepancy_type = ""
                        type_match = analysis.split("TYPE:", 1)
                        if len(type_match) > 1:
                            discrepancy_type = type_match[1].split("\n", 1)[0].strip()
                        
                        details = ""
                        details_match = analysis.split("DETAILS:", 1)
                        if len(details_match) > 1:
                            details = details_match[1].split("SEVERITY:", 1)[0].strip()
                        
                        # Default to YELLOW for fallback model
                        severity = "YELLOW"
                        severity_match = analysis.split("SEVERITY:", 1)
                        if len(severity_match) > 1:
                            severity_text = severity_match[1].strip()
                            if "RED" in severity_text:
                                severity = "RED"
                            elif "ORANGE" in severity_text:
                                severity = "ORANGE"
                        
                        # Add "Analyzed with fallback model" to details
                        if details:
                            details += " (Analyzed with fallback model)"
                        else:
                            details = "Discrepancy detected but details unclear. (Analyzed with fallback model)"
                        
                        differences.append({
                            "query": query,
                            "responses": doc_summaries,
                            "type": discrepancy_type if discrepancy_type else "Discrepancy type unclear",
                            "details": details,
                            "severity": severity,
                            "metadata": doc_metadata,
                            "text_samples": doc_text_samples,
                            "comparison_type": comparison_name
                        })
                        
                        print(f"    ⚠️ Discrepancy found using fallback model: {severity}")
                    else:
                        print(f"    ✓ No discrepancy found based on criteria (fallback model)")
                else:
                    print(f"    ⚠️ Warning: Failed to analyze with fallback model")
            except Exception as e:
                print(f"    ⚠️ Error analyzing with fallback model: {str(e)}")
    
    print(f"Found {len(differences)} potential discrepancies between Mosaic and {comparison_name}")
    return differences

def generate_potential_correction(diff: Dict) -> str:
    correction_prompt = f"""
    Based on these document responses to the query "{diff['query']}":
    
    SFC Mosaic Note: {diff['responses'].get("SFC Mosaic Note", "No data available")}
    {diff['comparison_type']}: {diff['responses'].get(diff['comparison_type'], "No data available")}
    
    Discrepancy type: {diff['type']}
    
    Provide a SPECIFIC, ACTIONABLE correction for the SFC Mosaic Note. Your recommendation must:
    1. Be precise and detailed (e.g., "Add missing revenue figure of ₹X for FY2023" or "Verify and correct the CEO name")
    2. Include specific values, dates, and details to be corrected
    3. If values need verification, specify exactly what needs to be verified
    4. Use at most 3 sentences - be concise and direct
    
    DO NOT simply say "Further investigation required" without specifying what exactly needs investigation.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=correction_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text'):
            correction = response.text.strip()
            
            if correction.lower() == "further investigation required." or correction.lower() == "further investigation required":
                return "Further investigation required - specifically, verify the correct information from the company's official financial records."
                
            return correction
        else:
            return "Verify data from official filings and update Mosaic report accordingly."
    except Exception as e:
        print(f"Error generating potential correction: {str(e)}")
        return "Verify data from official filings and update Mosaic report accordingly."

def generate_issue_title(diff: Dict, diff_idx: int) -> str:
    issue_title_prompt = f"""
    Generate a VERY CONCISE issue title (5-7 words) for this discrepancy:
    
    Query: {diff['query']}
    Details: {diff['details']}
    Type: {diff['type']}
    
    Follow these rules:
    1. Start with an action verb or key category (e.g., "Missing", "Inconsistent", "Extra", "Contradictory")
    2. Include the specific data element with discrepancy (e.g., "Revenue Figures", "Promoter Details", "CEO Name")
    3. Be specific - mention exactly what's different
    4. Use professional financial terminology
    5. Do NOT exceed a total of 7 words
    
    Examples:
    - "Missing Q3 Revenue Figures" 
    - "Undisclosed Promoter Shareholding Details"
    - "Contradictory CEO Background Information"
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=issue_title_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=50,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text'):
            title = response.text.strip()
            if title.startswith('"') and title.endswith('"'):
                title = title[1:-1]
            return title
        else:
            return f"Issue #{diff_idx+1}: {diff['query'][:30]}..."
    except Exception as e:
        print(f"Error generating issue title: {str(e)}")
        return f"Issue #{diff_idx+1}: {diff['query'][:30]}..."

def generate_report(drhp_differences: List[Dict], rating_differences: List[Dict], output_file: str):
    print(f"Generating report to {output_file}")
    
    total_drhp = len(drhp_differences)
    drhp_red = len([d for d in drhp_differences if d.get("severity") == "RED"])
    drhp_orange = len([d for d in drhp_differences if d.get("severity") == "ORANGE"])
    drhp_yellow = len([d for d in drhp_differences if d.get("severity") == "YELLOW"])
    
    total_rating = len(rating_differences)
    rating_red = len([d for d in rating_differences if d.get("severity") == "RED"])
    rating_orange = len([d for d in rating_differences if d.get("severity") == "ORANGE"])
    rating_yellow = len([d for d in rating_differences if d.get("severity") == "YELLOW"])
    
    summary_prompt = f"""
    Generate an executive summary for a document comparison analysis report with these findings:
    
    Mosaic Note vs DRHP:
    - Total discrepancies: {total_drhp}
    - RED (major) discrepancies: {drhp_red}
    - ORANGE (moderate) discrepancies: {drhp_orange}
    - YELLOW (minor) discrepancies: {drhp_yellow}
    
    Mosaic Note vs Credit Rating Report:
    - Total discrepancies: {total_rating}
    - RED (major) discrepancies: {rating_red}
    - ORANGE (moderate) discrepancies: {rating_orange}
    - YELLOW (minor) discrepancies: {rating_yellow}
    
    The executive summary should:
    1. Explain that the analysis was conducted iteratively (first Mosaic vs DRHP, then Mosaic vs Credit Rating)
    2. Clarify that discrepancies were only flagged when Mosaic included information not in the other document or missed information present in the other document
    3. Summarize the key findings in terms of severity and impact
    4. Provide high-level recommendations
    
    Keep the summary to 300-400 words, maintain a professional tone, and ensure it is COMPLETE with no cut-off sentences.
    """
    
    try:
        exec_summary = "## Executive Summary\n\n"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=summary_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text'):
            exec_summary += response.text.strip()
        else:
            exec_summary += "Failed to generate executive summary."
    except Exception as e:
        print(f"Error generating executive summary: {str(e)}")
        exec_summary += f"Error generating executive summary: {str(e)}"
    
    md_report = "# Document Comparison Analysis Report\n\n"
    md_report += exec_summary
    md_report += "\n\n## Analysis Summary\n\n"
    md_report += "### Mosaic Note vs DRHP Comparison\n"
    md_report += f"- **Total Discrepancies**: {total_drhp}\n"
    md_report += f"- **RED (Major)**: {drhp_red}\n"
    md_report += f"- **ORANGE (Moderate)**: {drhp_orange}\n"
    md_report += f"- **YELLOW (Minor)**: {drhp_yellow}\n\n"
    
    md_report += "### Mosaic Note vs Credit Rating Report Comparison\n"
    md_report += f"- **Total Discrepancies**: {total_rating}\n"
    md_report += f"- **RED (Major)**: {rating_red}\n"
    md_report += f"- **ORANGE (Moderate)**: {rating_orange}\n"
    md_report += f"- **YELLOW (Minor)**: {rating_yellow}\n\n"
    
    # Process DRHP differences
    all_drhp_issues = []
    
    for diff_idx, diff in enumerate(drhp_differences):
        issue_title = generate_issue_title(diff, diff_idx)
        potential_correction = generate_potential_correction(diff)
        
        page_number = "Not specified"
        mosaic_metadata = diff.get('metadata', {}).get('SFC Mosaic Note', {})
        if mosaic_metadata and 'page' in mosaic_metadata:
            page_number = mosaic_metadata.get('page', "Not specified")
        
        issue = {
            "title": issue_title,
            "type": diff['type'],
            "details": diff['details'],
            "mosaic_statement": diff['responses'].get("SFC Mosaic Note", "Not available"),
            "comparison_statement": diff['responses'].get(diff['comparison_type'], "Not available"),
            "correction": potential_correction,
            "page_number": page_number,
            "severity": diff.get("severity", "YELLOW")
        }
        
        all_drhp_issues.append(issue)
    
    # Process Rating differences
    all_rating_issues = []
    
    for diff_idx, diff in enumerate(rating_differences):
        issue_title = generate_issue_title(diff, diff_idx)
        potential_correction = generate_potential_correction(diff)
        
        page_number = "Not specified"
        mosaic_metadata = diff.get('metadata', {}).get('SFC Mosaic Note', {})
        if mosaic_metadata and 'page' in mosaic_metadata:
            page_number = mosaic_metadata.get('page', "Not specified")
        
        issue = {
            "title": issue_title,
            "type": diff['type'],
            "details": diff['details'],
            "mosaic_statement": diff['responses'].get("SFC Mosaic Note", "Not available"),
            "comparison_statement": diff['responses'].get(diff['comparison_type'], "Not available"),
            "correction": potential_correction,
            "page_number": page_number,
            "severity": diff.get("severity", "YELLOW")
        }
        
        all_rating_issues.append(issue)
    
    # Group DRHP issues by severity
    drhp_by_severity = {"RED": [], "ORANGE": [], "YELLOW": []}
    for issue in all_drhp_issues:
        severity = issue.get("severity", "YELLOW")
        drhp_by_severity[severity].append(issue)
    
    # Group Rating issues by severity
    rating_by_severity = {"RED": [], "ORANGE": [], "YELLOW": []}
    for issue in all_rating_issues:
        severity = issue.get("severity", "YELLOW")
        rating_by_severity[severity].append(issue)
    
    # Add DRHP comparison section
    md_report += "## Mosaic Note vs DRHP Comparison\n\n"
    
    if drhp_by_severity["RED"]:
        md_report += "### RED (Major) Discrepancies\n\n"
        for i, issue in enumerate(drhp_by_severity["RED"]):
            md_report += f"#### Issue {i+1}: {issue['title']}\n\n"
            md_report += f"**Type**: {issue['type']}\n\n"
            md_report += f"**Details**: {issue['details']}\n\n"
            md_report += f"**Mosaic Report Statement**: {issue['mosaic_statement']}\n\n"
            md_report += f"**DRHP Statement**: {issue['comparison_statement']}\n\n"
            md_report += f"**Potential Correction**: {issue['correction']}\n\n"
            md_report += f"**Page Number (Mosaic Report)**: {issue['page_number']}\n\n"
            md_report += "---\n\n"
    
    if drhp_by_severity["ORANGE"]:
        md_report += "### ORANGE (Moderate) Discrepancies\n\n"
        for i, issue in enumerate(drhp_by_severity["ORANGE"]):
            md_report += f"#### Issue {i+1}: {issue['title']}\n\n"
            md_report += f"**Type**: {issue['type']}\n\n"
            md_report += f"**Details**: {issue['details']}\n\n"
            md_report += f"**Mosaic Report Statement**: {issue['mosaic_statement']}\n\n"
            md_report += f"**DRHP Statement**: {issue['comparison_statement']}\n\n"
            md_report += f"**Potential Correction**: {issue['correction']}\n\n"
            md_report += f"**Page Number (Mosaic Report)**: {issue['page_number']}\n\n"
            md_report += "---\n\n"
    
    if drhp_by_severity["YELLOW"]:
        md_report += "### YELLOW (Minor) Discrepancies\n\n"
        for i, issue in enumerate(drhp_by_severity["YELLOW"]):
            md_report += f"#### Issue {i+1}: {issue['title']}\n\n"
            md_report += f"**Type**: {issue['type']}\n\n"
            md_report += f"**Details**: {issue['details']}\n\n"
            md_report += f"**Mosaic Report Statement**: {issue['mosaic_statement']}\n\n"
            md_report += f"**DRHP Statement**: {issue['comparison_statement']}\n\n"
            md_report += f"**Potential Correction**: {issue['correction']}\n\n"
            md_report += f"**Page Number (Mosaic Report)**: {issue['page_number']}\n\n"
            md_report += "---\n\n"
    
    # Add Rating comparison section
    md_report += "## Mosaic Note vs Credit Rating Report Comparison\n\n"
    
    if rating_by_severity["RED"]:
        md_report += "### RED (Major) Discrepancies\n\n"
        for i, issue in enumerate(rating_by_severity["RED"]):
            md_report += f"#### Issue {i+1}: {issue['title']}\n\n"
            md_report += f"**Type**: {issue['type']}\n\n"
            md_report += f"**Details**: {issue['details']}\n\n"
            md_report += f"**Mosaic Report Statement**: {issue['mosaic_statement']}\n\n"
            md_report += f"**Credit Rating Report Statement**: {issue['comparison_statement']}\n\n"
            md_report += f"**Potential Correction**: {issue['correction']}\n\n"
            md_report += f"**Page Number (Mosaic Report)**: {issue['page_number']}\n\n"
            md_report += "---\n\n"
    
    if rating_by_severity["ORANGE"]:
        md_report += "### ORANGE (Moderate) Discrepancies\n\n"
        for i, issue in enumerate(rating_by_severity["ORANGE"]):
            md_report += f"#### Issue {i+1}: {issue['title']}\n\n"
            md_report += f"**Type**: {issue['type']}\n\n"
            md_report += f"**Details**: {issue['details']}\n\n"
            md_report += f"**Mosaic Report Statement**: {issue['mosaic_statement']}\n\n"
            md_report += f"**Credit Rating Report Statement**: {issue['comparison_statement']}\n\n"
            md_report += f"**Potential Correction**: {issue['correction']}\n\n"
            md_report += f"**Page Number (Mosaic Report)**: {issue['page_number']}\n\n"
            md_report += "---\n\n"
    
    if rating_by_severity["YELLOW"]:
        md_report += "### YELLOW (Minor) Discrepancies\n\n"
        for i, issue in enumerate(rating_by_severity["YELLOW"]):
            md_report += f"#### Issue {i+1}: {issue['title']}\n\n"
            md_report += f"**Type**: {issue['type']}\n\n"
            md_report += f"**Details**: {issue['details']}\n\n"
            md_report += f"**Mosaic Report Statement**: {issue['mosaic_statement']}\n\n"
            md_report += f"**Credit Rating Report Statement**: {issue['comparison_statement']}\n\n"
            md_report += f"**Potential Correction**: {issue['correction']}\n\n"
            md_report += f"**Page Number (Mosaic Report)**: {issue['page_number']}\n\n"
            md_report += "---\n\n"
    
    with open(output_file, 'w') as f:
        f.write(md_report)
    
    pdf_output = output_file.replace('.md', '.pdf')
    convert_markdown_to_pdf(output_file, pdf_output, "Document Comparison Analysis Report")
    
    print(f"Markdown report saved to {output_file}")
    print(f"PDF report saved to {pdf_output}")
    
    csv_file = output_file.replace('.md', '.csv')
    rows = []
    
    for issue in all_drhp_issues:
        row = {
            'Comparison': 'Mosaic vs DRHP',
            'Issue Title': issue['title'],
            'Type': issue['type'],
            'Details': issue['details'],
            'Severity': issue['severity'],
            'Mosaic Statement': issue['mosaic_statement'],
            'Comparison Statement': issue['comparison_statement'],
            'Potential Correction': issue['correction'],
            'Page Number (Mosaic)': issue['page_number']
        }
        rows.append(row)
    
    for issue in all_rating_issues:
        row = {
            'Comparison': 'Mosaic vs Credit Rating',
            'Issue Title': issue['title'],
            'Type': issue['type'],
            'Details': issue['details'],
            'Severity': issue['severity'],
            'Mosaic Statement': issue['mosaic_statement'],
            'Comparison Statement': issue['comparison_statement'],
            'Potential Correction': issue['correction'],
            'Page Number (Mosaic)': issue['page_number']
        }
        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        print(f"CSV report saved to {csv_file}")
    
    return {
        'drhp_total': total_drhp,
        'drhp_red': drhp_red,
        'drhp_orange': drhp_orange,
        'drhp_yellow': drhp_yellow,
        'rating_total': total_rating,
        'rating_red': rating_red,
        'rating_orange': rating_orange,
        'rating_yellow': rating_yellow,
        'report_file': output_file,
        'pdf_file': pdf_output,
        'csv_file': csv_file
    }

def handle_api_errors(func):
    """Decorator to handle API errors and provide appropriate messages"""
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"API Error on attempt {attempt+1}/{max_retries}: {str(e)}")
                if attempt == max_retries - 1:
                    print("All retries failed. Using default/fallback approach.")
                    # Let the function handle fallbacks
                    return func(*args, **kwargs, use_fallback=True)
                print(f"Retrying in 2 seconds...")
                import time
                time.sleep(2)
    return wrapper

def main():
    parser = argparse.ArgumentParser(description="Document Comparison Analyzer")
    parser.add_argument("--mosaic", required=True, help="Path to the SFC Mosaic Note PDF")
    parser.add_argument("--drhp", required=True, help="Path to the DRHP PDF")
    parser.add_argument("--rating", required=True, help="Path to the Credit Rating Report PDF")
    parser.add_argument("--output", default="comparison_report.md", help="Output file for the report")
    parser.add_argument("--mosaic-store", default="vector_store_mosaic", help="Vector store directory for Mosaic note")
    parser.add_argument("--drhp-store", default="vector_store_drhp", help="Vector store directory for DRHP")
    parser.add_argument("--rating-store", default="vector_store_rating", help="Vector store directory for Rating report")
    parser.add_argument("--reprocess", action="store_true", help="Force reprocessing PDFs")
    parser.add_argument("--skip-topics", action="store_true", help="Skip topic extraction and use defaults")
    parser.add_argument("--max-queries", type=int, default=40, help="Maximum number of queries to run")
    
    args = parser.parse_args()
    
    print("\n=== LOADING VECTOR STORES ===\n")
    mosaic_store = OCRVectorStore(index_type="HNSW", chunk_size=8000, chunk_overlap=400)
    drhp_store = OCRVectorStore(index_type="HNSW", chunk_size=8000, chunk_overlap=400)
    rating_store = OCRVectorStore(index_type="HNSW", chunk_size=8000, chunk_overlap=400)
    
    try:
        if args.reprocess:
            print("\n=== PROCESSING DOCUMENTS ===\n")
            mosaic_store = process_document(args.mosaic, args.mosaic_store, args.reprocess)
            drhp_store = process_document(args.drhp, args.drhp_store, args.reprocess)
            rating_store = process_document(args.rating, args.rating_store, args.reprocess)
        else:
            print(f"Loading Mosaic vector store from {args.mosaic_store}")
            mosaic_store.load(args.mosaic_store)
            
            print(f"Loading DRHP vector store from {args.drhp_store}")
            drhp_store.load(args.drhp_store)
            
            print(f"Loading Credit Rating vector store from {args.rating_store}")
            rating_store.load(args.rating_store)
        
        # Extract topics or use defaults
        if args.skip_topics:
            print("\n=== USING DEFAULT TOPICS ===\n")
            topics = default_topics()
        else:
            print("\n=== EXTRACTING KEY TOPICS ===\n")
            topics = extract_key_topics(mosaic_store)
        
        print("\n=== GENERATING SPECIFIC QUERIES ===\n")
        queries = generate_specific_queries(topics)
        
        # Limit the number of queries if specified
        if args.max_queries and len(queries) > args.max_queries:
            print(f"Limiting to {args.max_queries} queries as specified")
            queries = queries[:args.max_queries]
        
        # First comparison: Mosaic vs DRHP
        print("\n=== COMPARING MOSAIC NOTE VS DRHP ===\n")
        drhp_query_results = query_documents(queries, mosaic_store, drhp_store, "DRHP")
        drhp_differences = analyze_document_differences(drhp_query_results, "DRHP")
        
        # Second comparison: Mosaic vs Credit Rating Report
        print("\n=== COMPARING MOSAIC NOTE VS CREDIT RATING REPORT ===\n")
        rating_query_results = query_documents(queries, mosaic_store, rating_store, "Credit Rating Report")
        rating_differences = analyze_document_differences(rating_query_results, "Credit Rating Report")
        
        print("\n=== GENERATING REPORT ===\n")
        report_stats = generate_report(drhp_differences, rating_differences, args.output)
        
        print("\n=== ANALYSIS COMPLETE ===\n")
        print("Mosaic Note vs DRHP Comparison:")
        print(f"- Total discrepancies: {report_stats['drhp_total']}")
        print(f"- RED (Major): {report_stats['drhp_red']}")
        print(f"- ORANGE (Moderate): {report_stats['drhp_orange']}")
        print(f"- YELLOW (Minor): {report_stats['drhp_yellow']}")
        
        print("\nMosaic Note vs Credit Rating Comparison:")
        print(f"- Total discrepancies: {report_stats['rating_total']}")
        print(f"- RED (Major): {report_stats['rating_red']}")
        print(f"- ORANGE (Moderate): {report_stats['rating_orange']}")
        print(f"- YELLOW (Minor): {report_stats['rating_yellow']}")
        
        print(f"\nMarkdown report saved to {report_stats['report_file']}")
        print(f"PDF report saved to {report_stats['pdf_file']}")
        print(f"CSV report saved to {report_stats['csv_file']}")
    
    except Exception as e:
        print(f"\n=== ERROR DURING EXECUTION ===\n")
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your input files and arguments and try again.")

if __name__ == "__main__":
    main()