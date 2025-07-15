import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')

class AnimatedCompanyAnalyzer:
    """
    Creates animated financial analysis for individual companies or sector comparisons.
    Perfect for YouTube content and showcasing data evolution over time.
    """
    
    def __init__(self):
        """Initialize the animated analyzer."""
        self.data_dir = Path("data")
        self.output_dir = Path("animations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Company colors
        self.company_colors = {
            # Tech
            'MSFT': '#00BCF2', 'GOOGL': '#4285F4', 'AAPL': '#007AFF', 
            'META': '#1877F2', 'AMZN': '#FF9900', 'NVDA': '#76B900',
            'CRM': '#00A1E0', 'SNOW': '#29B5E8', 'PLTR': '#101010',
            
            # Healthcare  
            'JNJ': '#CE0058', 'PFE': '#0093D0', 'UNH': '#002677',
            'ABBV': '#071D49', 'TMO': '#F05523', 'DHR': '#004B87',
            'ILMN': '#5CB3CC', 'REGN': '#7B68EE',
            
            # Defense
            'LMT': '#005DAA', 'RTX': '#0033A0', 'BA': '#0039A6',
            'NOC': '#003366', 'GD': '#1B365D', 'LHX': '#002F5F',
            'TXT': '#004B87'
        }
        
        self.sectors = ['tech', 'healthcare', 'defense']
    
    def load_company_data(self, ticker):
        """
        Load data for a specific company.
        
        Args:
            ticker (str): Company ticker symbol
            
        Returns:
            pd.DataFrame: Company financial data
        """
        # Find the company file across all sectors
        for sector in self.sectors:
            file_path = self.data_dir / sector / f"{ticker.lower()}_financial_data.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    print(f"✅ Loaded {ticker}: {len(df)} data points from {sector} sector")
                    return df
                except Exception as e:
                    print(f"❌ Error loading {ticker}: {e}")
                    return pd.DataFrame()
        
        print(f"❌ Data file not found for {ticker}")
        return pd.DataFrame()
    
    def create_revenue_growth_race(self, companies, save_path=None, duration=10):
        """
        Create an animated bar chart race showing revenue growth over time.
        
        Args:
            companies (list): List of company tickers to compare
            save_path (str, optional): Path to save the animation
            duration (int): Animation duration in seconds
        """
        print(f"🎬 Creating revenue growth race for: {', '.join(companies)}")
        
        # Load data for all companies
        all_data = {}
        for ticker in companies:
            df = self.load_company_data(ticker)
            if not df.empty:
                all_data[ticker] = df
        
        if len(all_data) < 2:
            print("❌ Need at least 2 companies with data for comparison")
            return
        
        # Prepare revenue data by quarter
        revenue_metrics = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax']
        quarterly_data = {}
        
        for ticker, df in all_data.items():
            revenue_data = df[df['metric'].isin(revenue_metrics)]
            if not revenue_data.empty:
                # Group by quarter and get latest revenue for each quarter
                revenue_data['quarter'] = revenue_data['date'].dt.to_period('Q')
                quarterly_revenue = revenue_data.groupby('quarter')['value'].last().reset_index()
                quarterly_revenue['date'] = quarterly_revenue['quarter'].dt.start_time
                quarterly_revenue['revenue_billions'] = quarterly_revenue['value'] / 1e9
                quarterly_data[ticker] = quarterly_revenue[['date', 'revenue_billions']]
        
        if not quarterly_data:
            print("❌ No revenue data found for animation")
            return
        
        # Find common date range
        all_dates = []
        for df in quarterly_data.values():
            all_dates.extend(df['date'].tolist())
        
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Create quarterly date range
        date_range = pd.date_range(start=min_date, end=max_date, freq='Q')
        
        # Prepare data for animation
        animation_data = []
        for date in date_range:
            frame_data = []
            for ticker in companies:
                if ticker in quarterly_data:
                    # Get revenue up to this date
                    company_data = quarterly_data[ticker][quarterly_data[ticker]['date'] <= date]
                    if not company_data.empty:
                        latest_revenue = company_data.iloc[-1]['revenue_billions']
                        frame_data.append({
                            'ticker': ticker,
                            'revenue': latest_revenue,
                            'date': date
                        })
            
            if frame_data:
                frame_df = pd.DataFrame(frame_data).sort_values('revenue', ascending=True)
                frame_df['rank'] = range(len(frame_df))
                animation_data.append((date, frame_df))
        
        if not animation_data:
            print("❌ No data available for animation")
            return
        
        # Create the animation
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        def animate(frame_idx):
            ax.clear()
            ax.set_facecolor('black')
            
            if frame_idx < len(animation_data):
                date, frame_df = animation_data[frame_idx]
                
                # Create horizontal bar chart
                bars = ax.barh(frame_df['rank'], frame_df['revenue'],
                              color=[self.company_colors.get(ticker, '#00FF00') 
                                    for ticker in frame_df['ticker']],
                              height=0.8, alpha=0.8)
                
                # Add company labels
                for i, (_, row) in enumerate(frame_df.iterrows()):
                    ax.text(row['revenue'] + max(frame_df['revenue']) * 0.01, 
                           i, f"{row['ticker']} (${row['revenue']:.1f}B)",
                           va='center', fontweight='bold', color='white', fontsize=12)
                
                # Styling
                ax.set_xlim(0, max(frame_df['revenue']) * 1.3)
                ax.set_ylim(-0.5, len(frame_df) - 0.5)
                ax.set_yticks([])
                ax.set_xlabel('Revenue (Billions USD)', fontweight='bold', color='white', fontsize=14)
                ax.set_title(f'Revenue Growth Race\n{date.strftime("%Y Q%q")}', 
                           fontweight='bold', color='white', fontsize=18, pad=20)
                
                # Style the axes
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3, color='white')
        
        # Create animation
        frames = len(animation_data)
        interval = (duration * 1000) // frames  # milliseconds per frame
        
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                     interval=interval, repeat=True, blit=False)
        
        if save_path:
            print(f"💾 Saving animation to {save_path} (this may take a few minutes)...")
            writer = animation.PillowWriter(fps=frames//duration)
            anim.save(save_path, writer=writer)
            print(f"✅ Animation saved: {save_path}")
        else:
            plt.show()
        
        return anim
    
    def create_financial_journey(self, ticker, save_path=None, duration=15):
        """
        Create an animated journey through a company's financial metrics over time.
        
        Args:
            ticker (str): Company ticker symbol
            save_path (str, optional): Path to save the animation
            duration (int): Animation duration in seconds
        """
        print(f"🎬 Creating financial journey for {ticker}")
        
        df = self.load_company_data(ticker)
        if df.empty:
            print(f"❌ No data found for {ticker}")
            return
        
        # Prepare key metrics over time
        key_metrics = {
            'Revenue': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax'],
            'Net Income': ['NetIncomeLoss'],
            'Net Margin': ['NetMargin'],
            'R&D Intensity': ['RDIntensity'],
            'ROE': ['ROE']
        }
        
        # Get annual data (10-K filings)
        annual_data = df[df['form'] == '10-K'].copy()
        
        # Prepare data for each metric
        metric_data = {}
        for metric_name, metric_tags in key_metrics.items():
            metric_df = annual_data[annual_data['metric'].isin(metric_tags)]
            if not metric_df.empty:
                # Group by year and get the latest value for each year
                metric_df['year'] = metric_df['date'].dt.year
                yearly_data = metric_df.groupby('year')['value'].last().reset_index()
                yearly_data['date'] = pd.to_datetime(yearly_data['year'], format='%Y')
                
                # Convert to appropriate units
                if metric_name == 'Revenue':
                    yearly_data['display_value'] = yearly_data['value'] / 1e9  # Billions
                    yearly_data['unit'] = 'Billions USD'
                elif metric_name == 'Net Income':
                    yearly_data['display_value'] = yearly_data['value'] / 1e9  # Billions
                    yearly_data['unit'] = 'Billions USD'
                elif metric_name in ['Net Margin', 'R&D Intensity', 'ROE']:
                    yearly_data['display_value'] = yearly_data['value'] * 100  # Percentage
                    yearly_data['unit'] = '%'
                else:
                    yearly_data['display_value'] = yearly_data['value']
                    yearly_data['unit'] = ''
                
                metric_data[metric_name] = yearly_data
        
        if not metric_data:
            print(f"❌ No suitable metrics found for {ticker}")
            return
        
        # Find common years
        all_years = []
        for df in metric_data.values():
            all_years.extend(df['year'].tolist())
        
        common_years = sorted(list(set(all_years)))
        
        # Create the animation
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#0a0a0a')
        
        # Create subplots for different metrics
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        axes = {
            'Revenue': fig.add_subplot(gs[0, 0]),
            'Net Income': fig.add_subplot(gs[0, 1]),
            'Net Margin': fig.add_subplot(gs[1, 0]),
            'R&D Intensity': fig.add_subplot(gs[1, 1]),
            'ROE': fig.add_subplot(gs[2, :])
        }
        
        # Set background color for all axes
        for ax in axes.values():
            ax.set_facecolor('#0a0a0a')
        
        def animate(frame_idx):
            if frame_idx < len(common_years):
                current_year = common_years[frame_idx]
                
                for metric_name, ax in axes.items():
                    ax.clear()
                    ax.set_facecolor('#0a0a0a')
                    
                    if metric_name in metric_data:
                        data = metric_data[metric_name]
                        # Get data up to current year
                        historical_data = data[data['year'] <= current_year]
                        
                        if not historical_data.empty:
                            # Create line plot with growing line
                            color = self.company_colors.get(ticker, '#00FF00')
                            
                            if len(historical_data) > 1:
                                ax.plot(historical_data['year'], historical_data['display_value'],
                                       color=color, linewidth=3, marker='o', markersize=8,
                                       markerfacecolor=color, markeredgecolor='white',
                                       markeredgewidth=2)
                            
                            # Highlight current year
                            current_data = historical_data[historical_data['year'] == current_year]
                            if not current_data.empty:
                                current_value = current_data.iloc[0]['display_value']
                                ax.scatter(current_year, current_value, s=200, color='red',
                                         edgecolor='white', linewidth=3, zorder=5)
                                
                                # Add value annotation
                                unit = current_data.iloc[0]['unit']
                                ax.annotate(f'{current_value:.1f}{unit}',
                                          (current_year, current_value),
                                          xytext=(10, 10), textcoords='offset points',
                                          fontsize=12, fontweight='bold', color='white',
                                          bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor=color, alpha=0.8))
                            
                            # Set axis limits and styling
                            ax.set_xlim(min(common_years) - 1, max(common_years) + 1)
                            
                            if len(historical_data) > 0:
                                y_min = min(data['display_value']) * 0.9
                                y_max = max(data['display_value']) * 1.1
                                ax.set_ylim(y_min, y_max)
                            
                            ax.set_title(f'{metric_name}', fontweight='bold', color='white', fontsize=14)
                            ax.set_xlabel('Year', fontweight='bold', color='white')
                            
                            unit = data.iloc[0]['unit'] if not data.empty else ''
                            ax.set_ylabel(f'{metric_name} ({unit})', fontweight='bold', color='white')
                            
                            # Style the axes
                            ax.spines['bottom'].set_color('white')
                            ax.spines['left'].set_color('white')
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.tick_params(colors='white')
                            ax.grid(True, alpha=0.3, color='white')
                
                # Add main title
                fig.suptitle(f'{ticker} Financial Journey - {current_year}', 
                           fontsize=24, fontweight='bold', color='white', y=0.95)
        
        # Create animation
        frames = len(common_years)
        interval = (duration * 1000) // frames
        
        anim = animation.FuncAnimation(fig, animate, frames=frames,
                                     interval=interval, repeat=True, blit=False)
        
        if save_path:
            print(f"💾 Saving animation to {save_path} (this may take a few minutes)...")
            writer = animation.PillowWriter(fps=max(1, frames//duration))
            anim.save(save_path, writer=writer)
            print(f"✅ Animation saved: {save_path}")
        else:
            plt.show()
        
        return anim
    
    def create_sector_evolution(self, sector, save_path=None, duration=12):
        """
        Create animated evolution of all companies in a sector.
        
        Args:
            sector (str): Sector name ('tech', 'healthcare', 'defense')
            save_path (str, optional): Path to save the animation
            duration (int): Animation duration in seconds
        """
        print(f"🎬 Creating {sector} sector evolution animation")
        
        sector_path = self.data_dir / sector
        if not sector_path.exists():
            print(f"❌ Sector directory not found: {sector_path}")
            return
        
        # Load all companies in sector
        csv_files = list(sector_path.glob("*_financial_data.csv"))
        companies = []
        
        for csv_file in csv_files:
            ticker = csv_file.stem.replace('_financial_data', '').upper()
            companies.append(ticker)
        
        if len(companies) < 2:
            print(f"❌ Need at least 2 companies in {sector} sector")
            return
        
        print(f"📊 Found {len(companies)} companies: {', '.join(companies)}")
        
        # Create revenue race animation for the sector
        output_path = save_path or self.output_dir / f"{sector}_evolution.gif"
        return self.create_revenue_growth_race(companies, output_path, duration)
    
    def create_metric_comparison_race(self, companies, metric_name, save_path=None, duration=10):
        """
        Create animated race for a specific financial metric.
        
        Args:
            companies (list): List of company tickers
            metric_name (str): Metric to compare ('NetMargin', 'ROE', 'RDIntensity', etc.)
            save_path (str, optional): Path to save the animation
            duration (int): Animation duration in seconds
        """
        print(f"🎬 Creating {metric_name} comparison race for: {', '.join(companies)}")
        
        # Load data for all companies
        all_data = {}
        for ticker in companies:
            df = self.load_company_data(ticker)
            if not df.empty:
                all_data[ticker] = df
        
        if len(all_data) < 2:
            print("❌ Need at least 2 companies with data for comparison")
            return
        
        # Prepare metric data by year
        annual_data = {}
        
        for ticker, df in all_data.items():
            metric_data = df[(df['metric'] == metric_name) & (df['form'] == '10-K')]
            if not metric_data.empty:
                metric_data['year'] = metric_data['date'].dt.year
                yearly_data = metric_data.groupby('year')['value'].last().reset_index()
                
                # Convert to percentage if needed
                if metric_name in ['NetMargin', 'ROE', 'RDIntensity', 'GrossMargin', 'OperatingMargin']:
                    yearly_data['display_value'] = yearly_data['value'] * 100
                    unit = '%'
                else:
                    yearly_data['display_value'] = yearly_data['value']
                    unit = ''
                
                yearly_data['unit'] = unit
                annual_data[ticker] = yearly_data
        
        if not annual_data:
            print(f"❌ No {metric_name} data found for animation")
            return
        
        # Find common years
        all_years = []
        for df in annual_data.values():
            all_years.extend(df['year'].tolist())
        
        common_years = sorted(list(set(all_years)))
        
        # Prepare animation data
        animation_data = []
        for year in common_years:
            frame_data = []
            for ticker in companies:
                if ticker in annual_data:
                    company_data = annual_data[ticker][annual_data[ticker]['year'] == year]
                    if not company_data.empty:
                        value = company_data.iloc[0]['display_value']
                        frame_data.append({
                            'ticker': ticker,
                            'value': value,
                            'year': year
                        })
            
            if frame_data:
                frame_df = pd.DataFrame(frame_data).sort_values('value', ascending=True)
                frame_df['rank'] = range(len(frame_df))
                animation_data.append((year, frame_df))
        
        if not animation_data:
            print(f"❌ No data available for {metric_name} animation")
            return
        
        # Create the animation
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        def animate(frame_idx):
            ax.clear()
            ax.set_facecolor('#1a1a2e')
            
            if frame_idx < len(animation_data):
                year, frame_df = animation_data[frame_idx]
                
                # Create horizontal bar chart
                bars = ax.barh(frame_df['rank'], frame_df['value'],
                              color=[self.company_colors.get(ticker, '#FF6B6B') 
                                    for ticker in frame_df['ticker']],
                              height=0.7, alpha=0.9)
                
                # Add company labels and values
                for i, (_, row) in enumerate(frame_df.iterrows()):
                    # Company label on the left
                    ax.text(-max(frame_df['value']) * 0.02, i, row['ticker'],
                           va='center', ha='right', fontweight='bold', 
                           color='white', fontsize=14)
                    
                    # Value label on the bar
                    unit = '%' if metric_name in ['NetMargin', 'ROE', 'RDIntensity', 'GrossMargin', 'OperatingMargin'] else ''
                    ax.text(row['value'] + max(frame_df['value']) * 0.01, i, 
                           f"{row['value']:.1f}{unit}",
                           va='center', fontweight='bold', color='white', fontsize=12)
                
                # Styling
                ax.set_xlim(-max(frame_df['value']) * 0.1, max(frame_df['value']) * 1.2)
                ax.set_ylim(-0.5, len(frame_df) - 0.5)
                ax.set_yticks([])
                ax.set_xlabel(f'{metric_name} ({unit})', fontweight='bold', color='white', fontsize=14)
                
                # Format title
                title_map = {
                    'NetMargin': 'Net Profit Margin',
                    'ROE': 'Return on Equity',
                    'RDIntensity': 'R&D Intensity',
                    'GrossMargin': 'Gross Margin',
                    'OperatingMargin': 'Operating Margin'
                }
                display_name = title_map.get(metric_name, metric_name)
                ax.set_title(f'{display_name} Comparison\n{year}', 
                           fontweight='bold', color='white', fontsize=18, pad=20)
                
                # Style the axes
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3, color='white', axis='x')
        
        # Create animation
        frames = len(animation_data)
        interval = (duration * 1000) // frames
        
        anim = animation.FuncAnimation(fig, animate, frames=frames,
                                     interval=interval, repeat=True, blit=False)
        
        if save_path:
            print(f"💾 Saving animation to {save_path} (this may take a few minutes)...")
            writer = animation.PillowWriter(fps=max(1, frames//duration))
            anim.save(save_path, writer=writer)
            print(f"✅ Animation saved: {save_path}")
        else:
            plt.show()
        
        return anim

def main():
    """Demo function showing how to create various animations."""
    print("🎬 ANIMATED COMPANY ANALYSIS TOOLKIT")
    print("=" * 50)
    
    analyzer = AnimatedCompanyAnalyzer()
    
    # Example animations
    examples = [
        "1. Revenue Growth Race (Tech Companies)",
        "2. Individual Company Financial Journey", 
        "3. Sector Evolution Animation",
        "4. Specific Metric Comparison Race",
        "5. Create all animations"
    ]
    
    print("\nChoose an animation to create:")
    for example in examples:
        print(f"   {example}")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        # Revenue race for tech companies
        tech_companies = ['MSFT', 'GOOGL', 'AAPL', 'META', 'AMZN']
        analyzer.create_revenue_growth_race(
            tech_companies, 
            save_path=analyzer.output_dir / "tech_revenue_race.gif"
        )
    
    elif choice == "2":
        # Individual company journey
        ticker = input("Enter company ticker (e.g., MSFT): ").strip().upper()
        analyzer.create_financial_journey(
            ticker,
            save_path=analyzer.output_dir / f"{ticker.lower()}_financial_journey.gif"
        )
    
    elif choice == "3":
        # Sector evolution
        sector = input("Enter sector (tech/healthcare/defense): ").strip().lower()
        analyzer.create_sector_evolution(
            sector,
            save_path=analyzer.output_dir / f"{sector}_evolution.gif"
        )
    
    elif choice == "4":
        # Metric comparison
        companies = input("Enter companies (comma-separated, e.g., MSFT,GOOGL,AAPL): ").strip().split(',')
        companies = [c.strip().upper() for c in companies]
        metric = input("Enter metric (NetMargin/ROE/RDIntensity): ").strip()
        analyzer.create_metric_comparison_race(
            companies, metric,
            save_path=analyzer.output_dir / f"{metric.lower()}_comparison.gif"
        )
    
    elif choice == "5":
        # Create multiple animations
        print("🚀 Creating sample animations...")
        
        # Tech revenue race
        analyzer.create_revenue_growth_race(
            ['MSFT', 'GOOGL', 'AAPL'], 
            save_path=analyzer.output_dir / "tech_revenue_race.gif"
        )
        
        # MSFT financial journey
        analyzer.create_financial_journey(
            'MSFT',
            save_path=analyzer.output_dir / "msft_financial_journey.gif"
        )
        
        # ROE comparison
        analyzer.create_metric_comparison_race(
            ['MSFT', 'GOOGL', 'AAPL'], 'ROE',
            save_path=analyzer.output_dir / "roe_comparison.gif"
        )
        
        print("🎉 All sample animations created!")
    
    print(f"\n✅ Animations saved in: {analyzer.output_dir}")
    print("💡 Perfect for YouTube content and presentations!")

if __name__ == "__main__":
    main()