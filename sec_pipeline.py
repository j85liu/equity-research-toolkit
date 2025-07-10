import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
from pathlib import Path
import numpy as np

# Import email from config file (keep this private)
try:
    from config import SEC_EMAIL
except ImportError:
    SEC_EMAIL = None

class SECDataPipeline:
    """
    A comprehensive SEC filing data pipeline for equity research.
    Retrieves and processes 10-K/10-Q filings for financial analysis.
    """
    
    def __init__(self, email=None):
        """
        Initialize the SEC data pipeline.
        
        Args:
            email (str, optional): Your email address (required by SEC API)
                                 Will try config.SEC_EMAIL, then SEC_API_EMAIL env var
        """
        if not email:
            # Try config file first, then environment variable
            email = SEC_EMAIL or os.getenv('SEC_API_EMAIL')
            if not email:
                raise ValueError("Email required. Add to config.py or set SEC_API_EMAIL environment variable")
        
        self.base_url = "https://data.sec.gov/api/xbrl"
        self.headers = {
            "User-Agent": f"equity-research-toolkit/1.0 ({email})"
        }
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Create sector subdirectories
        self.sectors = {
            'tech': Path("data/tech"),
            'healthcare': Path("data/healthcare"), 
            'defense': Path("data/defense")
        }
        
        for sector_dir in self.sectors.values():
            sector_dir.mkdir(exist_ok=True)
        
        # Key financial metrics to extract
        self.key_metrics = {
            # Income Statement
            'Revenues': 'us-gaap:Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax': 'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
            'CostOfRevenue': 'us-gaap:CostOfRevenue',
            'GrossProfit': 'us-gaap:GrossProfit',
            'OperatingIncomeLoss': 'us-gaap:OperatingIncomeLoss',
            'NetIncomeLoss': 'us-gaap:NetIncomeLoss',
            'EarningsPerShareBasic': 'us-gaap:EarningsPerShareBasic',
            'EarningsPerShareDiluted': 'us-gaap:EarningsPerShareDiluted',
            
            # Balance Sheet
            'Assets': 'us-gaap:Assets',
            'AssetsCurrent': 'us-gaap:AssetsCurrent',
            'Cash': 'us-gaap:Cash',
            'CashAndCashEquivalentsAtCarryingValue': 'us-gaap:CashAndCashEquivalentsAtCarryingValue',
            'Liabilities': 'us-gaap:Liabilities',
            'LiabilitiesCurrent': 'us-gaap:LiabilitiesCurrent',
            'LongTermDebt': 'us-gaap:LongTermDebt',
            'StockholdersEquity': 'us-gaap:StockholdersEquity',
            
            # Cash Flow Statement
            'NetCashProvidedByUsedInOperatingActivities': 'us-gaap:NetCashProvidedByUsedInOperatingActivities',
            'NetCashProvidedByUsedInInvestingActivities': 'us-gaap:NetCashProvidedByUsedInInvestingActivities',
            'NetCashProvidedByUsedInFinancingActivities': 'us-gaap:NetCashProvidedByUsedInFinancingActivities',
            'PaymentsToAcquirePropertyPlantAndEquipment': 'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment',
            
            # Additional Key Metrics
            'ResearchAndDevelopmentExpense': 'us-gaap:ResearchAndDevelopmentExpense',
            'WeightedAverageNumberOfSharesOutstandingBasic': 'us-gaap:WeightedAverageNumberOfSharesOutstandingBasic',
            'WeightedAverageNumberOfDilutedSharesOutstanding': 'us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding'
        }
    
    def get_company_cik(self, ticker):
        """
        Get the CIK (Central Index Key) for a company ticker.
        
        Args:
            ticker (str): Company ticker symbol
            
        Returns:
            str: CIK number (padded to 10 digits)
        """
        try:
            # Get company tickers JSON
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            companies = response.json()
            print(f"Looking for ticker: {ticker}")
            
            for company_data in companies.values():
                if company_data['ticker'].upper() == ticker.upper():
                    cik = str(company_data['cik_str']).zfill(10)
                    print(f"Found CIK for {ticker}: {cik}")
                    return cik
            
            print(f"Ticker {ticker} not found in SEC database")
            return None
            
        except Exception as e:
            print(f"Error getting CIK for {ticker}: {e}")
            return None
    
    def get_company_facts(self, cik):
        """
        Get all company facts for a given CIK.
        
        Args:
            cik (str): Company CIK
            
        Returns:
            dict: Company facts data
        """
        try:
            url = f"{self.base_url}/companyfacts/CIK{cik}.json"
            print(f"Fetching data from: {url}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            print(f"Successfully retrieved company facts for CIK {cik}")
            
            # Debug: print available fact categories
            if 'facts' in data:
                print(f"Available fact categories: {list(data['facts'].keys())}")
                if 'us-gaap' in data['facts']:
                    us_gaap_metrics = list(data['facts']['us-gaap'].keys())
                    print(f"Found {len(us_gaap_metrics)} US-GAAP metrics")
                    # Print first few metrics as examples
                    print(f"Sample metrics: {us_gaap_metrics[:5]}")
            
            return data
            
        except Exception as e:
            print(f"Error getting company facts for CIK {cik}: {e}")
            return None
    
    def extract_metric_data(self, facts_data, metric_tag):
        """
        Extract specific metric data from company facts.
        
        Args:
            facts_data (dict): Company facts data
            metric_tag (str): XBRL tag for the metric
            
        Returns:
            list: List of metric data points
        """
        try:
            if 'facts' not in facts_data:
                print(f"No 'facts' key found in data")
                return []
            
            us_gaap = facts_data['facts'].get('us-gaap', {})
            
            if metric_tag not in us_gaap:
                # Try without the 'us-gaap:' prefix
                clean_tag = metric_tag.replace('us-gaap:', '')
                if clean_tag not in us_gaap:
                    print(f"Metric {metric_tag} not found in US-GAAP data")
                    return []
                else:
                    metric_tag = clean_tag
            
            metric_data = us_gaap[metric_tag]
            units = metric_data.get('units', {})
            
            print(f"Found metric {metric_tag} with units: {list(units.keys())}")
            
            # Get USD data (most common)
            if 'USD' in units:
                data_points = units['USD']
                print(f"Found {len(data_points)} USD data points for {metric_tag}")
                return data_points
            
            # Fall back to other units if USD not available
            for unit_type, unit_data in units.items():
                if unit_data:
                    print(f"Using {unit_type} units for {metric_tag} ({len(unit_data)} data points)")
                    return unit_data
            
            return []
            
        except Exception as e:
            print(f"Error extracting metric {metric_tag}: {e}")
            return []
    
    def process_company_data(self, ticker, cik=None):
        """
        Process all financial data for a company.
        
        Args:
            ticker (str): Company ticker
            cik (str, optional): Company CIK (will be looked up if not provided)
            
        Returns:
            pd.DataFrame: Processed financial data
        """
        print(f"Processing data for {ticker}...")
        
        # Get CIK if not provided
        if not cik:
            cik = self.get_company_cik(ticker)
            if not cik:
                return pd.DataFrame()
        
        # Get company facts
        facts_data = self.get_company_facts(cik)
        if not facts_data:
            return pd.DataFrame()
        
        # Extract all metrics
        all_data = []
        
        for metric_name, metric_tag in self.key_metrics.items():
            metric_data = self.extract_metric_data(facts_data, metric_tag)
            
            for data_point in metric_data:
                # Filter for annual and quarterly data
                if data_point.get('form') in ['10-K', '10-Q']:
                    all_data.append({
                        'ticker': ticker,
                        'cik': cik,
                        'metric': metric_name,
                        'value': data_point.get('val'),
                        'date': data_point.get('end'),
                        'filed_date': data_point.get('filed'),
                        'form': data_point.get('form'),
                        'fiscal_year': data_point.get('fy'),
                        'fiscal_period': data_point.get('fp'),
                        'frame': data_point.get('frame')
                    })
        
        if not all_data:
            print(f"No data found for {ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Convert date columns
        df['date'] = pd.to_datetime(df['date'])
        df['filed_date'] = pd.to_datetime(df['filed_date'])
        
        # Sort by date
        df = df.sort_values(['date', 'filed_date'])
        
        # Remove duplicates (keep most recent filing for each date/metric)
        df = df.drop_duplicates(subset=['ticker', 'metric', 'date'], keep='last')
        
        print(f"Processed {len(df)} data points for {ticker}")
        return df
    
    def calculate_ratios(self, df):
        """
        Calculate financial ratios from the processed data.
        
        Args:
            df (pd.DataFrame): Processed financial data
            
        Returns:
            pd.DataFrame: Data with calculated ratios
        """
        # Pivot data for easier calculation
        pivot_df = df.pivot_table(
            index=['ticker', 'date', 'form', 'fiscal_year', 'fiscal_period'],
            columns='metric',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Calculate ratios
        ratios = []
        
        for _, row in pivot_df.iterrows():
            base_data = {
                'ticker': row['ticker'],
                'date': row['date'],
                'form': row['form'],
                'fiscal_year': row['fiscal_year'],
                'fiscal_period': row['fiscal_period']
            }
            
            # Revenue metrics
            revenue = row.get('Revenues') or row.get('RevenueFromContractWithCustomerExcludingAssessedTax')
            cost_of_revenue = row.get('CostOfRevenue')
            gross_profit = row.get('GrossProfit')
            
            if revenue and cost_of_revenue and revenue != 0:
                ratios.append({**base_data, 'metric': 'GrossMargin', 'value': (revenue - cost_of_revenue) / revenue})
            elif gross_profit and revenue and revenue != 0:
                ratios.append({**base_data, 'metric': 'GrossMargin', 'value': gross_profit / revenue})
            
            # Operating margin
            operating_income = row.get('OperatingIncomeLoss')
            if operating_income and revenue and revenue != 0:
                ratios.append({**base_data, 'metric': 'OperatingMargin', 'value': operating_income / revenue})
            
            # Net margin
            net_income = row.get('NetIncomeLoss')
            if net_income and revenue and revenue != 0:
                ratios.append({**base_data, 'metric': 'NetMargin', 'value': net_income / revenue})
            
            # R&D intensity
            rd_expense = row.get('ResearchAndDevelopmentExpense')
            if rd_expense and revenue and revenue != 0:
                ratios.append({**base_data, 'metric': 'RDIntensity', 'value': rd_expense / revenue})
            
            # Asset turnover
            assets = row.get('Assets')
            if assets and revenue and assets != 0:
                ratios.append({**base_data, 'metric': 'AssetTurnover', 'value': revenue / assets})
            
            # Current ratio
            current_assets = row.get('AssetsCurrent')
            current_liabilities = row.get('LiabilitiesCurrent')
            if current_assets and current_liabilities and current_liabilities != 0:
                ratios.append({**base_data, 'metric': 'CurrentRatio', 'value': current_assets / current_liabilities})
            
            # Debt to equity
            long_term_debt = row.get('LongTermDebt')
            stockholders_equity = row.get('StockholdersEquity')
            if long_term_debt and stockholders_equity and stockholders_equity != 0:
                ratios.append({**base_data, 'metric': 'DebtToEquity', 'value': long_term_debt / stockholders_equity})
            
            # ROE
            if net_income and stockholders_equity and stockholders_equity != 0:
                ratios.append({**base_data, 'metric': 'ROE', 'value': net_income / stockholders_equity})
        
        # Convert ratios to DataFrame and combine with original data
        if ratios:
            ratios_df = pd.DataFrame(ratios)
            combined_df = pd.concat([df, ratios_df], ignore_index=True)
        else:
            combined_df = df.copy()
        
        return combined_df
    
    def save_data(self, df, filename, sector=None):
        """
        Save processed data to CSV.
        
        Args:
            df (pd.DataFrame): Data to save
            filename (str): Filename for the CSV
            sector (str, optional): Sector subdirectory ('tech', 'healthcare', 'defense')
        """
        if sector and sector in self.sectors:
            filepath = self.sectors[sector] / filename
        else:
            filepath = self.data_dir / filename
            
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def process_multiple_companies(self, tickers, save_individual=True, save_combined=True):
        """
        Process multiple companies and optionally save the data.
        
        Args:
            tickers (list): List of ticker symbols
            save_individual (bool): Save individual company files
            save_combined (bool): Save combined file
            
        Returns:
            pd.DataFrame: Combined data for all companies
        """
        all_data = []
        
        for ticker in tickers:
            print(f"\n{'='*50}")
            print(f"Processing {ticker}")
            print(f"{'='*50}")
            
            try:
                # Process company data
                df = self.process_company_data(ticker)
                
                if df.empty:
                    print(f"No data retrieved for {ticker}")
                    continue
                
                # Calculate ratios
                df_with_ratios = self.calculate_ratios(df)
                
                # Save individual file if requested
                if save_individual:
                    filename = f"{ticker.lower()}_financial_data.csv"
                    self.save_data(df_with_ratios, filename)
                
                all_data.append(df_with_ratios)
                
                # Rate limiting - be respectful to SEC servers
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            if save_combined:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"combined_financial_data_{timestamp}.csv"
                self.save_data(combined_df, filename)
            
            print(f"\n🎉 SUCCESS: Combined dataset created with {len(combined_df)} total data points")
            return combined_df
        else:
            print("❌ No data was processed successfully")
            return pd.DataFrame()
    
    def process_all_sectors(self, tech_companies, healthcare_companies, defense_companies):
        """
        Process all sectors and return combined results.
        
        Args:
            tech_companies (list): List of tech company tickers
            healthcare_companies (list): List of healthcare company tickers
            defense_companies (list): List of defense company tickers
            
        Returns:
            dict: Dictionary with sector data and combined dataset
        """
        results = {}
        
        print("\n" + "="*60)
        print("🚀 PROCESSING ALL SECTORS")
        print("="*60)
        
        # Process Tech Sector
        print(f"\n📱 TECH SECTOR ({len(tech_companies)} companies)")
        print("-" * 40)
        tech_data = self.process_multiple_companies(tech_companies, sector='tech')
        results['tech'] = tech_data
        
        # Process Healthcare Sector
        print(f"\n🏥 HEALTHCARE SECTOR ({len(healthcare_companies)} companies)")
        print("-" * 40)
        healthcare_data = self.process_multiple_companies(healthcare_companies, sector='healthcare')
        results['healthcare'] = healthcare_data
        
        # Process Defense Sector
        print(f"\n🛡️ DEFENSE SECTOR ({len(defense_companies)} companies)")
        print("-" * 40)
        defense_data = self.process_multiple_companies(defense_companies, sector='defense')
        results['defense'] = defense_data
        
        # Combine all sectors
        all_sector_data = []
        for sector_name, sector_data in results.items():
            if not sector_data.empty:
                all_sector_data.append(sector_data)
        
        if all_sector_data:
            combined_all = pd.concat(all_sector_data, ignore_index=True)
            results['all_combined'] = combined_all
            
            # Print final summary (no master file saved)
            print(f"\n🎉 FINAL SUMMARY")
            print("=" * 60)
            print(f"📊 Total companies processed: {len(combined_all['ticker'].unique())}")
            print(f"📈 Total data points: {len(combined_all):,}")
            print(f"📅 Date range: {combined_all['date'].min().strftime('%Y-%m-%d')} to {combined_all['date'].max().strftime('%Y-%m-%d')}")
            print(f"🏢 Companies by sector:")
            
            for sector in ['tech', 'healthcare', 'defense']:
                if sector in results and not results[sector].empty:
                    companies = results[sector]['ticker'].unique()
                    print(f"   {sector.upper()}: {', '.join(companies)}")
            
            print(f"📁 Individual files saved in sector subdirectories:")
            print(f"   - data/tech/ ({len([x for x in results.get('tech', pd.DataFrame())['ticker'].unique() if x])} companies)")
            print(f"   - data/healthcare/ ({len([x for x in results.get('healthcare', pd.DataFrame())['ticker'].unique() if x])} companies)") 
            print(f"   - data/defense/ ({len([x for x in results.get('defense', pd.DataFrame())['ticker'].unique() if x])} companies)")
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the pipeline (will use config.SEC_EMAIL)
    try:
        pipeline = SECDataPipeline()
        print(f"✅ Using email from config file")
    except ValueError as e:
        print(f"❗ {e}")
        email = input("Please enter your email address for SEC API: ").strip()
        if not email:
            print("❌ Email is required. Exiting...")
            exit(1)
        pipeline = SECDataPipeline(email=email)
    
    # Define companies to analyze by sector
    tech_companies = ['MSFT', 'GOOGL', 'AAPL', 'META', 'AMZN', 'NVDA', 'CRM', 'SNOW', 'PLTR']
    healthcare_companies = ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'ILMN', 'REGN']
    defense_companies = ['LMT', 'RTX', 'BA', 'NOC', 'GD', 'LHX', 'TXT']
    
    print("🚀 EQUITY RESEARCH TOOLKIT - SEC DATA PIPELINE")
    print("=" * 60)
    print(f"📱 Tech companies: {len(tech_companies)} ({', '.join(tech_companies)})")
    print(f"🏥 Healthcare companies: {len(healthcare_companies)} ({', '.join(healthcare_companies)})")
    print(f"🛡️ Defense companies: {len(defense_companies)} ({', '.join(defense_companies)})")
    print(f"📊 Total companies to process: {len(tech_companies + healthcare_companies + defense_companies)}")
    
    # Ask user if they want to proceed with full processing
    print(f"\n⚠️ WARNING: This will process {len(tech_companies + healthcare_companies + defense_companies)} companies.")
    print("This may take 10-15 minutes and will make many API calls to the SEC.")
    proceed = input("Do you want to proceed? (y/N): ").strip().lower()
    
    if proceed not in ['y', 'yes']:
        print("📝 Running with just a few companies for testing...")
        tech_companies = tech_companies[:2]  # Just MSFT and GOOGL
        healthcare_companies = healthcare_companies[:2]  # Just JNJ and PFE
        defense_companies = defense_companies[:1]  # Just LMT
        print(f"🔬 Test run: {tech_companies + healthcare_companies + defense_companies}")
    
    # Process all sectors
    results = pipeline.process_all_sectors(tech_companies, healthcare_companies, defense_companies)
    
    # Optional: Quick analysis of results
    if 'all_combined' in results:
        print(f"\n📈 QUICK ANALYSIS:")
        print("-" * 30)
        all_data = results['all_combined']
        
        # Top metrics by data availability
        print(f"🔍 Top metrics by data points:")
        metric_counts = all_data['metric'].value_counts().head(10)
        for metric, count in metric_counts.items():
            print(f"   {metric}: {count:,} data points")
        
        # Companies with most data
        print(f"\n🏢 Companies with most data points:")
        company_counts = all_data['ticker'].value_counts().head(10)
        for company, count in company_counts.items():
            print(f"   {company}: {count:,} data points")
        
        print(f"\n✅ Data collection complete! Ready for visualization and analysis.")
        print(f"💡 Next steps:")
        print(f"   1. Run visualization toolkit on sector data")
        print(f"   2. Create YouTube content with peer comparisons")
        print(f"   3. Build investment thesis and analysis")
    else:
        print("❌ No data was successfully processed. Check error messages above.")