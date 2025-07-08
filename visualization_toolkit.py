import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class EquityVisualizationToolkit:
    """
    Professional visualization toolkit for equity research analysis.
    Creates publication-ready charts for financial analysis and YouTube content.
    """
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize the visualization toolkit.
        
        Args:
            figsize (tuple): Default figure size for charts
        """
        self.figsize = figsize
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#6C757D'
        }
        
        # Professional color palette
        self.company_colors = {
            'MSFT': '#00BCF2',
            'GOOGL': '#4285F4',
            'AAPL': '#007AFF',
            'META': '#1877F2',
            'AMZN': '#FF9900',
            'NVDA': '#76B900',
            'TSLA': '#CC0000'
        }
    
    def load_data(self, filepath_or_df):
        """
        Load financial data from CSV or DataFrame.
        
        Args:
            filepath_or_df: Path to CSV file or pandas DataFrame
            
        Returns:
            pd.DataFrame: Loaded and preprocessed data
        """
        if isinstance(filepath_or_df, str):
            df = pd.read_csv(filepath_or_df)
        else:
            df = filepath_or_df.copy()
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values(['ticker', 'date'])
        
        return df
    
    def plot_revenue_trends(self, df, companies=None, save_path=None):
        """
        Create revenue trend analysis chart.
        
        Args:
            df (pd.DataFrame): Financial data
            companies (list, optional): List of companies to include
            save_path (str, optional): Path to save the chart
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Filter for revenue data
        revenue_metrics = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax']
        revenue_data = df[df['metric'].isin(revenue_metrics)].copy()
        
        if companies:
            revenue_data = revenue_data[revenue_data['ticker'].isin(companies)]
        
        # Group by company and date, take the first revenue metric available
        revenue_pivot = revenue_data.groupby(['ticker', 'date'])['value'].first().reset_index()
        
        # Plot 1: Absolute Revenue Trends
        for ticker in revenue_pivot['ticker'].unique():
            company_data = revenue_pivot[revenue_pivot['ticker'] == ticker]
            color = self.company_colors.get(ticker, self.colors['primary'])
            
            ax1.plot(company_data['date'], company_data['value'] / 1e9, 
                    marker='o', linewidth=2.5, label=ticker, color=color, markersize=6)
        
        ax1.set_title('Revenue Trends - Absolute Values', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Revenue (Billions USD)', fontsize=12, fontweight='bold')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Revenue Growth Rates (YoY)
        for ticker in revenue_pivot['ticker'].unique():
            company_data = revenue_pivot[revenue_pivot['ticker'] == ticker].copy()
            company_data = company_data.sort_values('date')
            
            # Calculate YoY growth
            company_data['yoy_growth'] = company_data['value'].pct_change(periods=4) * 100
            
            color = self.company_colors.get(ticker, self.colors['primary'])
            ax2.plot(company_data['date'], company_data['yoy_growth'], 
                    marker='s', linewidth=2.5, label=ticker, color=color, markersize=6)
        
        ax2.set_title('Revenue Growth Rates (Year-over-Year)', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('YoY Growth (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_profitability_analysis(self, df, companies=None, save_path=None):
        """
        Create comprehensive profitability analysis.
        
        Args:
            df (pd.DataFrame): Financial data
            companies (list, optional): List of companies to include
            save_path (str, optional): Path to save the chart
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        if companies:
            df = df[df['ticker'].isin(companies)]
        
        # Get the latest data for each company
        latest_data = df.loc[df.groupby(['ticker', 'metric'])['date'].idxmax()]
        
        # Prepare margin data
        margin_metrics = ['GrossMargin', 'OperatingMargin', 'NetMargin']
        margin_data = latest_data[latest_data['metric'].isin(margin_metrics)]
        
        if not margin_data.empty:
            # Plot 1: Current Margins Comparison
            margin_pivot = margin_data.pivot(index='ticker', columns='metric', values='value')
            margin_pivot = margin_pivot * 100  # Convert to percentage
            
            margin_pivot.plot(kind='bar', ax=axes[0,0], width=0.8)
            axes[0,0].set_title('Current Profitability Margins', fontsize=14, fontweight='bold')
            axes[0,0].set_ylabel('Margin (%)', fontsize=12, fontweight='bold')
            axes[0,0].legend(title='Margin Type')
            axes[0,0].tick_params(axis='x', rotation=45)
            axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Margin Trends Over Time
        margin_trends = df[df['metric'].isin(margin_metrics)].copy()
        if not margin_trends.empty:
            for ticker in margin_trends['ticker'].unique():
                for metric in margin_metrics:
                    data = margin_trends[(margin_trends['ticker'] == ticker) & 
                                       (margin_trends['metric'] == metric)]
                    if not data.empty:
                        color = self.company_colors.get(ticker, self.colors['primary'])
                        linestyle = '-' if metric == 'NetMargin' else '--' if metric == 'OperatingMargin' else ':'
                        axes[0,1].plot(data['date'], data['value'] * 100, 
                                     label=f"{ticker} {metric}", color=color, linestyle=linestyle)
            
            axes[0,1].set_title('Margin Trends Over Time', fontsize=14, fontweight='bold')
            axes[0,1].set_ylabel('Margin (%)', fontsize=12, fontweight='bold')
            axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: R&D Intensity
        rd_data = latest_data[latest_data['metric'] == 'RDIntensity']
        if not rd_data.empty:
            bars = axes[1,0].bar(rd_data['ticker'], rd_data['value'] * 100,
                               color=[self.company_colors.get(ticker, self.colors['primary']) 
                                     for ticker in rd_data['ticker']])
            axes[1,0].set_title('R&D Intensity (R&D/Revenue)', fontsize=14, fontweight='bold')
            axes[1,0].set_ylabel('R&D Intensity (%)', fontsize=12, fontweight='bold')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: ROE Comparison
        roe_data = latest_data[latest_data['metric'] == 'ROE']
        if not roe_data.empty:
            bars = axes[1,1].bar(roe_data['ticker'], roe_data['value'] * 100,
                               color=[self.company_colors.get(ticker, self.colors['accent']) 
                                     for ticker in roe_data['ticker']])
            axes[1,1].set_title('Return on Equity (ROE)', fontsize=14, fontweight='bold')
            axes[1,1].set_ylabel('ROE (%)', fontsize=12, fontweight='bold')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Comprehensive Profitability Analysis', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_financial_health_dashboard(self, df, companies=None, save_path=None):
        """
        Create financial health dashboard.
        
        Args:
            df (pd.DataFrame): Financial data
            companies (list, optional): List of companies to include
            save_path (str, optional): Path to save the chart
        """
        fig = plt.figure(figsize=(16, 12))
        
        if companies:
            df = df[df['ticker'].isin(companies)]
        
        # Get latest data
        latest_data = df.loc[df.groupby(['ticker', 'metric'])['date'].idxmax()]
        
        # Create subplots with custom layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Current Ratio (Liquidity)
        ax1 = fig.add_subplot(gs[0, 0])
        current_ratio_data = latest_data[latest_data['metric'] == 'CurrentRatio']
        if not current_ratio_data.empty:
            bars = ax1.bar(current_ratio_data['ticker'], current_ratio_data['value'],
                          color=[self.company_colors.get(ticker, self.colors['primary']) 
                                for ticker in current_ratio_data['ticker']])
            ax1.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Healthy (>2.0)')
            ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Risk (<1.0)')
            ax1.set_title('Current Ratio', fontweight='bold')
            ax1.set_ylabel('Ratio')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Debt-to-Equity
        ax2 = fig.add_subplot(gs[0, 1])
        debt_equity_data = latest_data[latest_data['metric'] == 'DebtToEquity']
        if not debt_equity_data.empty:
            bars = ax2.bar(debt_equity_data['ticker'], debt_equity_data['value'],
                          color=[self.company_colors.get(ticker, self.colors['secondary']) 
                                for ticker in debt_equity_data['ticker']])
            ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Conservative (<0.5)')
            ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Moderate (1.0)')
            ax2.set_title('Debt-to-Equity Ratio', fontweight='bold')
            ax2.set_ylabel('Ratio')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cash Position
        ax3 = fig.add_subplot(gs[0, 2])
        cash_data = latest_data[latest_data['metric'] == 'CashAndCashEquivalentsAtCarryingValue']
        if not cash_data.empty:
            bars = ax3.bar(cash_data['ticker'], cash_data['value'] / 1e9,
                          color=[self.company_colors.get(ticker, self.colors['accent']) 
                                for ticker in cash_data['ticker']])
            ax3.set_title('Cash Position', fontweight='bold')
            ax3.set_ylabel('Cash (Billions USD)')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Revenue vs Expenses Trends
        ax4 = fig.add_subplot(gs[1, :])
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker]
            
            # Revenue
            revenue_data = ticker_data[ticker_data['metric'].isin(['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax'])]
            revenue_grouped = revenue_data.groupby('date')['value'].first().reset_index()
            
            # Operating expenses (approximated by Revenue - Operating Income)
            op_income_data = ticker_data[ticker_data['metric'] == 'OperatingIncomeLoss']
            
            color = self.company_colors.get(ticker, self.colors['primary'])
            
            if not revenue_grouped.empty:
                ax4.plot(revenue_grouped['date'], revenue_grouped['value'] / 1e9,
                        label=f"{ticker} Revenue", linewidth=3, color=color)
            
            if not op_income_data.empty:
                ax4.plot(op_income_data['date'], op_income_data['value'] / 1e9,
                        label=f"{ticker} Operating Income", linewidth=2, color=color, linestyle='--')
        
        ax4.set_title('Revenue vs Operating Income Trends', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Amount (Billions USD)')
        ax4.set_xlabel('Date')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Asset Efficiency
        ax5 = fig.add_subplot(gs[2, :2])
        asset_turnover_data = latest_data[latest_data['metric'] == 'AssetTurnover']
        if not asset_turnover_data.empty:
            bars = ax5.bar(asset_turnover_data['ticker'], asset_turnover_data['value'],
                          color=[self.company_colors.get(ticker, self.colors['neutral']) 
                                for ticker in asset_turnover_data['ticker']])
            ax5.set_title('Asset Turnover (Revenue/Assets)', fontweight='bold')
            ax5.set_ylabel('Asset Turnover Ratio')
            ax5.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 6: Financial Summary Table
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        # Create summary metrics table
        summary_metrics = ['NetMargin', 'ROE', 'CurrentRatio', 'DebtToEquity']
        summary_data = latest_data[latest_data['metric'].isin(summary_metrics)]
        
        if not summary_data.empty:
            summary_pivot = summary_data.pivot(index='ticker', columns='metric', values='value')
            
            # Format the data for display
            if 'NetMargin' in summary_pivot.columns:
                summary_pivot['NetMargin'] = (summary_pivot['NetMargin'] * 100).round(1)
            if 'ROE' in summary_pivot.columns:
                summary_pivot['ROE'] = (summary_pivot['ROE'] * 100).round(1)
            if 'CurrentRatio' in summary_pivot.columns:
                summary_pivot['CurrentRatio'] = summary_pivot['CurrentRatio'].round(2)
            if 'DebtToEquity' in summary_pivot.columns:
                summary_pivot['DebtToEquity'] = summary_pivot['DebtToEquity'].round(2)
            
            # Create table
            table = ax6.table(cellText=summary_pivot.values,
                            rowLabels=summary_pivot.index,
                            colLabels=['Current\nRatio', 'Debt/Equity', 'Net Margin\n(%)', 'ROE\n(%)'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2)
            ax6.set_title('Key Metrics Summary', fontweight='bold', pad=20)
        
        plt.suptitle('Financial Health Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_quarterly_performance(self, df, ticker, save_path=None):
        """
        Create detailed quarterly performance analysis for a single company.
        
        Args:
            df (pd.DataFrame): Financial data
            ticker (str): Company ticker to analyze
            save_path (str, optional): Path to save the chart
        """
        company_data = df[df['ticker'] == ticker].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get quarterly data (10-Q forms)
        quarterly_data = company_data[company_data['form'] == '10-Q']
        
        # Plot 1: Quarterly Revenue Growth
        revenue_data = quarterly_data[quarterly_data['metric'].isin(['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax'])]
        revenue_grouped = revenue_data.groupby('date')['value'].first().reset_index()
        revenue_grouped = revenue_grouped.sort_values('date')
        revenue_grouped['qoq_growth'] = revenue_grouped['value'].pct_change() * 100
        
        color = self.company_colors.get(ticker, self.colors['primary'])
        
        if not revenue_grouped.empty:
            axes[0,0].bar(range(len(revenue_grouped)), revenue_grouped['value'] / 1e9, 
                         color=color, alpha=0.7)
            axes[0,0].set_title(f'{ticker} - Quarterly Revenue', fontweight='bold')
            axes[0,0].set_ylabel('Revenue (Billions USD)')
            axes[0,0].set_xticks(range(len(revenue_grouped)))
            axes[0,0].set_xticklabels([d.strftime('%Y-Q%q') for d in revenue_grouped['date']], rotation=45)
            axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Quarterly Growth Rates
        if not revenue_grouped.empty and len(revenue_grouped) > 1:
            axes[0,1].plot(range(1, len(revenue_grouped)), revenue_grouped['qoq_growth'][1:], 
                          marker='o', linewidth=3, color=color, markersize=8)
            axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0,1].set_title(f'{ticker} - Quarterly Growth (QoQ)', fontweight='bold')
            axes[0,1].set_ylabel('Growth Rate (%)')
            axes[0,1].set_xticks(range(1, len(revenue_grouped)))
            axes[0,1].set_xticklabels([d.strftime('%Y-Q%q') for d in revenue_grouped['date'][1:]], rotation=45)
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Quarterly Margins
        margin_metrics = ['GrossMargin', 'OperatingMargin', 'NetMargin']
        margin_data = quarterly_data[quarterly_data['metric'].isin(margin_metrics)]
        
        if not margin_data.empty:
            for metric in margin_metrics:
                metric_data = margin_data[margin_data['metric'] == metric]
                if not metric_data.empty:
                    linestyle = '-' if metric == 'NetMargin' else '--' if metric == 'OperatingMargin' else ':'
                    axes[1,0].plot(metric_data['date'], metric_data['value'] * 100, 
                                  label=metric, linewidth=2.5, linestyle=linestyle, marker='o')
            
            axes[1,0].set_title(f'{ticker} - Quarterly Margins', fontweight='bold')
            axes[1,0].set_ylabel('Margin (%)')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Cash Flow Analysis
        cash_flow_metrics = ['NetCashProvidedByUsedInOperatingActivities',
                           'NetCashProvidedByUsedInInvestingActivities',
                           'NetCashProvidedByUsedInFinancingActivities']
        
        cf_data = quarterly_data[quarterly_data['metric'].isin(cash_flow_metrics)]
        
        if not cf_data.empty:
            cf_pivot = cf_data.pivot(index='date', columns='metric', values='value')
            cf_pivot = cf_pivot / 1e9  # Convert to billions
            
            cf_pivot.plot(kind='bar', ax=axes[1,1], width=0.8)
            axes[1,1].set_title(f'{ticker} - Quarterly Cash Flows', fontweight='bold')
            axes[1,1].set_ylabel('Cash Flow (Billions USD)')
            axes[1,1].legend(['Operating', 'Investing', 'Financing'], loc='upper right')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        plt.suptitle(f'{ticker} - Quarterly Performance Analysis', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_peer_comparison(self, df, companies, save_path=None):
        """
        Create comprehensive peer comparison analysis.
        
        Args:
            df (pd.DataFrame): Financial data
            companies (list): List of company tickers to compare
            save_path (str, optional): Path to save the chart
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Filter data for specified companies
        peer_data = df[df['ticker'].isin(companies)]
        latest_data = peer_data.loc[peer_data.groupby(['ticker', 'metric'])['date'].idxmax()]
        
        # 1. Revenue Comparison (Last 12 Months)
        revenue_data = latest_data[latest_data['metric'].isin(['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax'])]
        if not revenue_data.empty:
            bars = axes[0,0].bar(revenue_data['ticker'], revenue_data['value'] / 1e9,
                               color=[self.company_colors.get(ticker, self.colors['primary']) 
                                     for ticker in revenue_data['ticker']])
            axes[0,0].set_title('Revenue Comparison (TTM)', fontweight='bold')
            axes[0,0].set_ylabel('Revenue (Billions USD)')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                             f'${height:.0f}B', ha='center', va='bottom', fontweight='bold')
        
        # 2. Profitability Comparison
        profitability_metrics = ['GrossMargin', 'OperatingMargin', 'NetMargin']
        prof_data = latest_data[latest_data['metric'].isin(profitability_metrics)]
        
        if not prof_data.empty:
            prof_pivot = prof_data.pivot(index='ticker', columns='metric', values='value')
            prof_pivot = prof_pivot * 100  # Convert to percentage
            
            prof_pivot.plot(kind='bar', ax=axes[0,1], width=0.8)
            axes[0,1].set_title('Profitability Margins', fontweight='bold')
            axes[0,1].set_ylabel('Margin (%)')
            axes[0,1].legend(title='Margin Type')
            axes[0,1].tick_params(axis='x', rotation=45)
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. R&D Intensity
        rd_data = latest_data[latest_data['metric'] == 'RDIntensity']
        if not rd_data.empty:
            bars = axes[0,2].bar(rd_data['ticker'], rd_data['value'] * 100,
                               color=[self.company_colors.get(ticker, self.colors['accent']) 
                                     for ticker in rd_data['ticker']])
            axes[0,2].set_title('R&D Intensity', fontweight='bold')
            axes[0,2].set_ylabel('R&D as % of Revenue')
            axes[0,2].grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Financial Health Metrics
        health_metrics = ['CurrentRatio', 'DebtToEquity']
        health_data = latest_data[latest_data['metric'].isin(health_metrics)]
        
        if not health_data.empty:
            health_pivot = health_data.pivot(index='ticker', columns='metric', values='value')
            health_pivot.plot(kind='bar', ax=axes[1,0], width=0.8)
            axes[1,0].set_title('Financial Health Ratios', fontweight='bold')
            axes[1,0].set_ylabel('Ratio')
            axes[1,0].legend(title='Metric')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. Return on Equity
        roe_data = latest_data[latest_data['metric'] == 'ROE']
        if not roe_data.empty:
            bars = axes[1,1].bar(roe_data['ticker'], roe_data['value'] * 100,
                               color=[self.company_colors.get(ticker, self.colors['success']) 
                                     for ticker in roe_data['ticker']])
            axes[1,1].set_title('Return on Equity', fontweight='bold')
            axes[1,1].set_ylabel('ROE (%)')
            axes[1,1].grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Asset Efficiency
        efficiency_data = latest_data[latest_data['metric'] == 'AssetTurnover']
        if not efficiency_data.empty:
            bars = axes[1,2].bar(efficiency_data['ticker'], efficiency_data['value'],
                               color=[self.company_colors.get(ticker, self.colors['neutral']) 
                                     for ticker in efficiency_data['ticker']])
            axes[1,2].set_title('Asset Turnover', fontweight='bold')
            axes[1,2].set_ylabel('Revenue / Assets')
            axes[1,2].grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[1,2].text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                             f'{height:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Peer Comparison Analysis', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Usage Examples and Demo Functions
def demo_visualizations():
    """
    Demo function showing how to use the visualization toolkit.
    """
    print("🎨 Equity Research Visualization Toolkit Demo")
    print("=" * 50)
    
    # Initialize the toolkit
    viz = EquityVisualizationToolkit()
    
    # Load sample data (replace with your actual data files)
    print("📊 Loading sample data...")
    
    # Example usage with actual data files
    try:
        # Load your generated data
        msft_data = viz.load_data('data/msft_financial_data.csv')
        googl_data = viz.load_data('data/googl_financial_data.csv')
        
        # Combine datasets
        combined_data = pd.concat([msft_data, googl_data], ignore_index=True)
        
        print(f"✅ Loaded data: {len(combined_data)} total data points")
        print(f"   Companies: {', '.join(combined_data['ticker'].unique())}")
        print(f"   Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
        
        # Create visualizations
        print("\n🔍 Creating visualizations...")
        
        # 1. Revenue Trends
        print("   → Revenue trends analysis...")
        viz.plot_revenue_trends(combined_data, save_path='outputs/revenue_trends.png')
        
        # 2. Profitability Analysis
        print("   → Profitability analysis...")
        viz.plot_profitability_analysis(combined_data, save_path='outputs/profitability_analysis.png')
        
        # 3. Financial Health Dashboard
        print("   → Financial health dashboard...")
        viz.plot_financial_health_dashboard(combined_data, save_path='outputs/financial_health.png')
        
        # 4. Individual Company Deep Dive
        print("   → MSFT quarterly performance...")
        viz.plot_quarterly_performance(combined_data, 'MSFT', save_path='outputs/msft_quarterly.png')
        
        # 5. Peer Comparison
        print("   → Peer comparison analysis...")
        viz.create_peer_comparison(combined_data, ['MSFT', 'GOOGL'], save_path='outputs/peer_comparison.png')
        
        print("\n🎉 All visualizations created successfully!")
        print("   Check the 'outputs/' folder for saved charts")
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("   Make sure you've run the SEC pipeline first to generate data files")
        
        # Create sample data for demonstration
        print("\n📝 Creating sample data for demonstration...")
        sample_data = create_sample_data()
        
        print("   → Sample revenue trends...")
        viz.plot_revenue_trends(sample_data)
        
        print("   → Sample profitability analysis...")
        viz.plot_profitability_analysis(sample_data)

def create_sample_data():
    """
    Create sample financial data for demonstration purposes.
    """
    np.random.seed(42)
    
    companies = ['MSFT', 'GOOGL', 'AAPL']
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='Q')
    
    data = []
    
    for company in companies:
        base_revenue = np.random.uniform(50e9, 150e9)  # Base revenue
        growth_rate = np.random.uniform(0.05, 0.15)   # Annual growth rate
        
        for i, date in enumerate(dates):
            # Revenue with growth and seasonality
            revenue = base_revenue * (1 + growth_rate) ** (i/4) * (1 + 0.1 * np.sin(i * np.pi / 2))
            
            # Add some noise
            revenue *= np.random.uniform(0.95, 1.05)
            
            # Generate related metrics
            gross_margin = np.random.uniform(0.6, 0.8)
            operating_margin = np.random.uniform(0.2, 0.4)
            net_margin = np.random.uniform(0.15, 0.25)
            
            # Add data points
            metrics = {
                'Revenues': revenue,
                'GrossMargin': gross_margin,
                'OperatingMargin': operating_margin,
                'NetMargin': net_margin,
                'RDIntensity': np.random.uniform(0.1, 0.2),
                'CurrentRatio': np.random.uniform(1.5, 3.0),
                'DebtToEquity': np.random.uniform(0.1, 0.5),
                'ROE': np.random.uniform(0.15, 0.35),
                'AssetTurnover': np.random.uniform(0.3, 0.8)
            }
            
            for metric, value in metrics.items():
                data.append({
                    'ticker': company,
                    'date': date,
                    'metric': metric,
                    'value': value,
                    'form': '10-Q' if date.month % 12 != 0 else '10-K'
                })
    
    return pd.DataFrame(data)

# YouTube Content Creation Helper
class YouTubeContentCreator:
    """
    Helper class for creating YouTube-ready financial analysis content.
    """
    
    def __init__(self, viz_toolkit):
        """
        Initialize with a visualization toolkit.
        
        Args:
            viz_toolkit: EquityVisualizationToolkit instance
        """
        self.viz = viz_toolkit
    
    def create_company_spotlight(self, df, ticker, output_dir='youtube_content'):
        """
        Create a complete company spotlight package for YouTube.
        
        Args:
            df (pd.DataFrame): Financial data
            ticker (str): Company ticker
            output_dir (str): Directory to save content
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎬 Creating YouTube content for {ticker}")
        
        # 1. Quarterly Performance Deep Dive
        self.viz.plot_quarterly_performance(
            df, ticker, 
            save_path=f'{output_dir}/{ticker.lower()}_quarterly_analysis.png'
        )
        
        # 2. Create investment thesis summary
        company_data = df[df['ticker'] == ticker]
        latest_data = company_data.loc[company_data.groupby('metric')['date'].idxmax()]
        
        # Generate key talking points
        talking_points = self.generate_talking_points(latest_data, ticker)
        
        # Save talking points
        with open(f'{output_dir}/{ticker.lower()}_talking_points.txt', 'w') as f:
            f.write(f"📊 {ticker} Investment Analysis - Key Talking Points\n")
            f.write("=" * 50 + "\n\n")
            for point in talking_points:
                f.write(f"• {point}\n")
        
        print(f"✅ Content created in {output_dir}/")
        return talking_points
    
    def generate_talking_points(self, latest_data, ticker):
        """
        Generate key talking points for video content.
        
        Args:
            latest_data (pd.DataFrame): Latest financial data for company
            ticker (str): Company ticker
            
        Returns:
            list: List of talking points
        """
        points = []
        
        # Revenue analysis
        revenue_data = latest_data[latest_data['metric'].isin(['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax'])]
        if not revenue_data.empty:
            revenue = revenue_data['value'].iloc[0] / 1e9
            points.append(f"Current revenue run rate: ${revenue:.1f}B annually")
        
        # Profitability
        net_margin = latest_data[latest_data['metric'] == 'NetMargin']
        if not net_margin.empty:
            margin = net_margin['value'].iloc[0] * 100
            points.append(f"Net profit margin: {margin:.1f}% - {'Strong' if margin > 20 else 'Moderate' if margin > 10 else 'Concerning'}")
        
        # R&D Investment
        rd_intensity = latest_data[latest_data['metric'] == 'RDIntensity']
        if not rd_intensity.empty:
            rd = rd_intensity['value'].iloc[0] * 100
            points.append(f"R&D intensity: {rd:.1f}% of revenue - {'Innovation-focused' if rd > 15 else 'Moderate innovation investment'}")
        
        # Financial health
        current_ratio = latest_data[latest_data['metric'] == 'CurrentRatio']
        if not current_ratio.empty:
            cr = current_ratio['value'].iloc[0]
            health = 'Excellent' if cr > 2 else 'Good' if cr > 1.5 else 'Concerning'
            points.append(f"Liquidity position: {health} (Current ratio: {cr:.1f})")
        
        # ROE
        roe = latest_data[latest_data['metric'] == 'ROE']
        if not roe.empty:
            roe_val = roe['value'].iloc[0] * 100
            points.append(f"Return on equity: {roe_val:.1f}% - {'Excellent' if roe_val > 25 else 'Good' if roe_val > 15 else 'Average'}")
        
        return points
    
    def create_sector_comparison_video(self, df, companies, sector_name, output_dir='youtube_content'):
        """
        Create sector comparison content for YouTube.
        
        Args:
            df (pd.DataFrame): Financial data
            companies (list): List of company tickers
            sector_name (str): Name of the sector
            output_dir (str): Directory to save content
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎬 Creating {sector_name} sector comparison content")
        
        # Create peer comparison chart
        self.viz.create_peer_comparison(
            df, companies,
            save_path=f'{output_dir}/{sector_name.lower()}_sector_comparison.png'
        )
        
        # Create talking points for sector
        sector_points = self.generate_sector_talking_points(df, companies, sector_name)
        
        # Save sector analysis
        with open(f'{output_dir}/{sector_name.lower()}_sector_analysis.txt', 'w') as f:
            f.write(f"📊 {sector_name} Sector Analysis\n")
            f.write("=" * 50 + "\n\n")
            for point in sector_points:
                f.write(f"• {point}\n")
        
        print(f"✅ {sector_name} sector content created in {output_dir}/")
        return sector_points
    
    def generate_sector_talking_points(self, df, companies, sector_name):
        """
        Generate sector-level talking points.
        
        Args:
            df (pd.DataFrame): Financial data
            companies (list): List of company tickers
            sector_name (str): Sector name
            
        Returns:
            list: Sector talking points
        """
        points = []
        
        sector_data = df[df['ticker'].isin(companies)]
        latest_data = sector_data.loc[sector_data.groupby(['ticker', 'metric'])['date'].idxmax()]
        
        # Revenue leader
        revenue_data = latest_data[latest_data['metric'].isin(['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax'])]
        if not revenue_data.empty:
            revenue_leader = revenue_data.loc[revenue_data['value'].idxmax()]
            points.append(f"Revenue leader: {revenue_leader['ticker']} with ${revenue_leader['value']/1e9:.1f}B")
        
        # Profitability comparison
        margin_data = latest_data[latest_data['metric'] == 'NetMargin']
        if not margin_data.empty:
            best_margin = margin_data.loc[margin_data['value'].idxmax()]
            worst_margin = margin_data.loc[margin_data['value'].idxmin()]
            points.append(f"Profitability range: {best_margin['ticker']} leads at {best_margin['value']*100:.1f}%, {worst_margin['ticker']} trails at {worst_margin['value']*100:.1f}%")
        
        # R&D spending
        rd_data = latest_data[latest_data['metric'] == 'RDIntensity']
        if not rd_data.empty:
            rd_leader = rd_data.loc[rd_data['value'].idxmax()]
            points.append(f"Innovation leader: {rd_leader['ticker']} invests {rd_leader['value']*100:.1f}% of revenue in R&D")
        
        # Financial health
        health_data = latest_data[latest_data['metric'] == 'CurrentRatio']
        if not health_data.empty:
            healthiest = health_data.loc[health_data['value'].idxmax()]
            points.append(f"Strongest balance sheet: {healthiest['ticker']} with current ratio of {healthiest['value']:.1f}")
        
        return points

if __name__ == "__main__":
    # Run the demo
    demo_visualizations()