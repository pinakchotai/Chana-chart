import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

app = Flask(__name__)

# Function to calculate trading indicators
def calculate_indicators(df, date_col, price_col):
    # Create a copy of the dataframe to avoid modifying the original
    df_indicators = df.copy()
    
    # 1. Simple Moving Averages (SMA)
    df_indicators['SMA_20'] = df_indicators[price_col].rolling(window=20).mean()
    df_indicators['SMA_50'] = df_indicators[price_col].rolling(window=50).mean()
    
    # 2. Exponential Moving Average (EMA)
    df_indicators['EMA_20'] = df_indicators[price_col].ewm(span=20, adjust=False).mean()
    
    # 3. Relative Strength Index (RSI)
    delta = df_indicators[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df_indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. Bollinger Bands
    sma_20 = df_indicators[price_col].rolling(window=20).mean()
    std_20 = df_indicators[price_col].rolling(window=20).std()
    df_indicators['BB_Upper'] = sma_20 + (std_20 * 2)
    df_indicators['BB_Lower'] = sma_20 - (std_20 * 2)
    
    # 5. MACD (Moving Average Convergence Divergence)
    df_indicators['EMA_12'] = df_indicators[price_col].ewm(span=12, adjust=False).mean()
    df_indicators['EMA_26'] = df_indicators[price_col].ewm(span=26, adjust=False).mean()
    df_indicators['MACD'] = df_indicators['EMA_12'] - df_indicators['EMA_26']
    df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()
    df_indicators['MACD_Histogram'] = df_indicators['MACD'] - df_indicators['MACD_Signal']
    
    return df_indicators

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    try:
        # Load Excel file
        excel_path = os.path.join(os.path.dirname(__file__), 'CHANA R&D.xlsx')
        
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        # Try to identify the date column
        date_col = None
        for col in df.columns:
            # Check if column name contains 'date'
            if 'date' in str(col).lower():
                date_col = col
                break
        
        # If date column not found by name, try to find a column with date values
        if date_col is None:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]) or \
                   (df[col].dtype == 'object' and pd.to_datetime(df[col], errors='coerce').notna().all()):
                    date_col = col
                    break
        
        # Try to identify the price column
        price_col = None
        price_keywords = ['price', 'rate', 'value', 'spot']
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in price_keywords):
                price_col = col
                break
        
        # If price column not found by name, try to find a numeric column
        if price_col is None:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and col != date_col:
                    price_col = col
                    break
        
        # If we still don't have the columns, use the first two columns
        if date_col is None:
            date_col = df.columns[0]
        if price_col is None:
            price_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # Ensure date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Drop rows with NaN dates
        df = df.dropna(subset=[date_col])
        
        # Ensure price column is numeric
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Sort by date
        df = df.sort_values(by=date_col)
        
        # Generate synthetic OHLC data for candlestick charts
        # We'll create reasonable variations based on the close price
        df['open'] = df[price_col].shift(1).fillna(df[price_col])
        
        # Create a random factor for high/low variation (between 0.5% and 2%)
        np.random.seed(42)  # For reproducibility
        variation_factors = np.random.uniform(0.005, 0.02, size=len(df))
        
        # Calculate high and low values
        price_values = df[price_col].values
        open_values = df['open'].values
        
        # High is the maximum of open and close, plus a small random variation
        df['high'] = np.maximum(open_values, price_values) * (1 + variation_factors)
        
        # Low is the minimum of open and close, minus a small random variation
        df['low'] = np.minimum(open_values, price_values) * (1 - variation_factors)
        
        # Ensure high is always >= both open and close
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df[price_col]))
        
        # Ensure low is always <= both open and close
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df[price_col]))
        
        # Calculate indicators
        df_indicators = calculate_indicators(df, date_col, price_col)
        
        # Copy OHLC columns to the indicators dataframe
        df_indicators['open'] = df['open']
        df_indicators['high'] = df['high']
        df_indicators['low'] = df['low']
        
        # Convert date to string for JSON serialization
        df_indicators['date_str'] = df_indicators[date_col].dt.strftime('%Y-%m-%d')
        
        # Create a dictionary for JSON response
        data = {
            'date': df_indicators['date_str'].tolist(),
            'price': df_indicators[price_col].tolist(),
            'open': df_indicators['open'].tolist(),
            'high': df_indicators['high'].tolist(),
            'low': df_indicators['low'].tolist(),
            'sma_20': df_indicators['SMA_20'].tolist(),
            'sma_50': df_indicators['SMA_50'].tolist(),
            'ema_20': df_indicators['EMA_20'].tolist(),
            'bb_upper': df_indicators['BB_Upper'].tolist(),
            'bb_lower': df_indicators['BB_Lower'].tolist(),
            'rsi': df_indicators['RSI'].tolist(),
            'macd': df_indicators['MACD'].tolist(),
            'macd_signal': df_indicators['MACD_Signal'].tolist(),
            'macd_histogram': df_indicators['MACD_Histogram'].tolist(),
            'date_col': date_col,
            'price_col': price_col
        }
        
        return jsonify(data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plot')
def plot():
    try:
        # Get indicator choices from query parameters
        indicators = request.args.get('indicators', 'none')
        chart_type = request.args.get('chart_type', 'line')
        
        # Debug logging
        print(f"Request received - indicators: {indicators}, chart_type: {chart_type}")
        
        # Load Excel file
        excel_path = os.path.join(os.path.dirname(__file__), 'CHANA R&D.xlsx')
        
        # Check if file exists
        if not os.path.exists(excel_path):
            print(f"Error: Excel file not found at {excel_path}")
            return jsonify({'error': 'Data file not found'}), 404
        
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        # Try to identify the date column
        date_col = None
        for col in df.columns:
            # Check if column name contains 'date'
            if 'date' in str(col).lower():
                date_col = col
                break
        
        # If date column not found by name, try to find a column with date values
        if date_col is None:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]) or \
                   (df[col].dtype == 'object' and pd.to_datetime(df[col], errors='coerce').notna().all()):
                    date_col = col
                    break
        
        # Try to identify the price column
        price_col = None
        price_keywords = ['price', 'rate', 'value', 'spot']
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in price_keywords):
                price_col = col
                break
        
        # If price column not found by name, try to find a numeric column
        if price_col is None:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and col != date_col:
                    price_col = col
                    break
        
        # If we still don't have the columns, use the first two columns
        if date_col is None:
            date_col = df.columns[0]
        if price_col is None:
            price_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # Ensure date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Drop rows with NaN dates
        df = df.dropna(subset=[date_col])
        
        # Ensure price column is numeric
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Sort by date
        df = df.sort_values(by=date_col)
        
        # Generate synthetic OHLC data for candlestick charts
        df['open'] = df[price_col].shift(1).fillna(df[price_col])
        
        # Create a random factor for high/low variation (between 0.5% and 2%)
        np.random.seed(42)  # For reproducibility
        variation_factors = np.random.uniform(0.005, 0.02, size=len(df))
        
        # Calculate high and low values
        price_values = df[price_col].values
        open_values = df['open'].values
        
        # High is the maximum of open and close, plus a small random variation
        df['high'] = np.maximum(open_values, price_values) * (1 + variation_factors)
        
        # Low is the minimum of open and close, minus a small random variation
        df['low'] = np.minimum(open_values, price_values) * (1 - variation_factors)
        
        # Ensure high is always >= both open and close
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df[price_col]))
        
        # Ensure low is always <= both open and close
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df[price_col]))
        
        # Calculate indicators
        df_indicators = calculate_indicators(df, date_col, price_col)
        
        # Copy OHLC columns
        df_indicators['open'] = df['open']
        df_indicators['high'] = df['high']
        df_indicators['low'] = df['low']
        
        # Check the chart type
        print(f"Creating {chart_type} chart...")
        
        # Create the Plotly figure with subplots if indicators are requested
        if indicators != 'none':
            # Use more space for the main chart when indicators are shown and increase vertical spacing
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.15,  # Increased spacing between subplots
                               row_heights=[0.72, 0.28],  # Adjusted heights for better separation
                               subplot_titles=('Chickpeas Spot Price', 'Indicators'))
            
            # Add main price trace - either line or candlestick
            if chart_type == 'candlestick':
                fig.add_trace(go.Candlestick(
                    x=df_indicators[date_col],
                    open=df_indicators['open'],
                    high=df_indicators['high'],
                    low=df_indicators['low'],
                    close=df_indicators[price_col],
                    name='OHLC',
                    increasing=dict(line=dict(color='#26a69a', width=1.5), fillcolor='rgba(38, 166, 154, 0.7)'),
                    decreasing=dict(line=dict(color='#ef5350', width=1.5), fillcolor='rgba(239, 83, 80, 0.7)'),
                    whiskerwidth=0.8,
                    line=dict(width=1)
                ), row=1, col=1)
                print("Added enhanced candlestick trace")
            else:
                # Default to line chart with improved styling
                fig.add_trace(go.Scatter(
                    x=df_indicators[date_col], 
                    y=df_indicators[price_col],
                    mode='lines',
                    name='Price',
                    line=dict(color='#2c3e50', width=2.5, shape='spline'),
                    opacity=0.9
                ), row=1, col=1)
            
            # Add SMA 20 with improved styling
            fig.add_trace(go.Scatter(
                x=df_indicators[date_col], 
                y=df_indicators['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#3498db', width=1.8, dash='solid'),
                opacity=0.8
            ), row=1, col=1)
            
            # Add SMA 50 with improved styling
            fig.add_trace(go.Scatter(
                x=df_indicators[date_col], 
                y=df_indicators['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='#e74c3c', width=1.8, dash='solid'),
                opacity=0.8
            ), row=1, col=1)
            
            # Add Bollinger Bands with improved styling
            fig.add_trace(go.Scatter(
                x=df_indicators[date_col], 
                y=df_indicators['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(46, 204, 113, 0.9)', width=1.5, dash='dash'),
                fill=None
            ), row=1, col=1)
            
            # Add BB Middle (usually SMA 20) if available
            if 'BB_Middle' in df_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=df_indicators[date_col], 
                    y=df_indicators['BB_Middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='rgba(52, 152, 219, 0.6)', width=1, dash='dot'),
                    opacity=0.6
                ), row=1, col=1)
            
            # Add BB Lower with fill for the band area
            fig.add_trace(go.Scatter(
                x=df_indicators[date_col], 
                y=df_indicators['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(46, 204, 113, 0.9)', width=1.5, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(46, 204, 113, 0.1)'
            ), row=1, col=1)
            
            # Add either RSI or MACD to the bottom panel
            if indicators == 'rsi':
                # Add RSI Line with improved styling
                fig.add_trace(go.Scatter(
                    x=df_indicators[date_col], 
                    y=df_indicators['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='#2980b9', width=2.5)
                ), row=2, col=1)
                
                # Add overbought line (70)
                fig.add_shape(
                    type="line", 
                    x0=df_indicators[date_col].iloc[0], 
                    x1=df_indicators[date_col].iloc[-1],
                    y0=70, 
                    y1=70, 
                    line=dict(
                        color="rgba(231, 76, 60, 0.8)", 
                        width=1.5, 
                        dash="dash"
                    ),
                    row=2, col=1
                )
                
                # Add middle line (50)
                fig.add_shape(
                    type="line", 
                    x0=df_indicators[date_col].iloc[0], 
                    x1=df_indicators[date_col].iloc[-1],
                    y0=50, 
                    y1=50, 
                    line=dict(
                        color="rgba(150, 150, 150, 0.8)", 
                        width=1.5, 
                        dash="dash"
                    ),
                    row=2, col=1
                )
                
                # Add oversold line (30)
                fig.add_shape(
                    type="line", 
                    x0=df_indicators[date_col].iloc[0], 
                    x1=df_indicators[date_col].iloc[-1],
                    y0=30, 
                    y1=30, 
                    line=dict(
                        color="rgba(46, 204, 113, 0.8)", 
                        width=1.5, 
                        dash="dash"
                    ),
                    row=2, col=1
                )
                
                # Add light background coloring for overbought area
                fig.add_shape(
                    type="rect",
                    x0=df_indicators[date_col].iloc[0],
                    x1=df_indicators[date_col].iloc[-1],
                    y0=70,
                    y1=100,
                    fillcolor="rgba(231, 76, 60, 0.1)",
                    line=dict(width=0),
                    row=2, col=1
                )
                
                # Add light background coloring for oversold area
                fig.add_shape(
                    type="rect",
                    x0=df_indicators[date_col].iloc[0],
                    x1=df_indicators[date_col].iloc[-1],
                    y0=0,
                    y1=30,
                    fillcolor="rgba(46, 204, 113, 0.1)",
                    line=dict(width=0),
                    row=2, col=1
                )
                
                # Set background for RSI panel slightly different for better distinction
                fig.update_layout(
                    plot_bgcolor='rgba(250, 250, 250, 1)',
                )
                
                # Improve RSI y-axis formatting and title
                fig.update_yaxes(
                    title_text="RSI", 
                    title_font=dict(size=12, color="#333"),
                    row=2, 
                    col=1,
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.2)',
                    zeroline=False,
                    range=[0, 100],
                    fixedrange=True  # Prevents y-axis zooming for cleaner interaction
                )
                
                # Set different background color for the indicator panel
                fig.update_layout(
                    plot_bgcolor='rgba(248, 249, 250, 1)'
                )
                
                # Add different background for the indicator subplot
                fig.add_shape(
                    type="rect",
                    x0=0,
                    x1=1,
                    y0=0,
                    y1=0.28,
                    fillcolor="rgba(240, 245, 250, 0.8)",
                    line=dict(width=0),
                    xref="paper",
                    yref="paper",
                    layer="below"
                )
                
            else:  # Default to MACD
                # Add MACD Line
                fig.add_trace(go.Scatter(
                    x=df_indicators[date_col], 
                    y=df_indicators['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='#2980b9', width=2.5)
                ), row=2, col=1)
                
                # Add MACD Signal Line
                fig.add_trace(go.Scatter(
                    x=df_indicators[date_col], 
                    y=df_indicators['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='#e74c3c', width=2)
                ), row=2, col=1)
                
                # Add MACD Histogram with improved styling
                colors = ['rgba(39, 174, 96, 0.5)' if val >= 0 else 'rgba(192, 57, 43, 0.5)' for val in df_indicators['MACD_Histogram']]
                
                fig.add_trace(go.Bar(
                    x=df_indicators[date_col], 
                    y=df_indicators['MACD_Histogram'],
                    name='Histogram',
                    marker_color=colors,
                    marker_line_width=0,
                    opacity=0.7,
                    width=24*60*60*1000 * 0.8  # 80% of a day in milliseconds for better spacing
                ), row=2, col=1)
                
                # Add zero line for reference with improved styling
                fig.add_shape(
                    type="line", 
                    x0=df_indicators[date_col].iloc[0], 
                    x1=df_indicators[date_col].iloc[-1],
                    y0=0, 
                    y1=0, 
                    line=dict(
                        color="rgba(150, 150, 150, 0.8)", 
                        width=1.5, 
                        dash="dash"
                    ),
                    row=2, col=1
                )
                
                # Set background for MACD panel slightly different for better distinction
                fig.update_layout(
                    plot_bgcolor='rgba(250, 250, 250, 1)',
                )
                
                # Improve MACD y-axis formatting and title
                fig.update_yaxes(
                    title_text="MACD", 
                    title_font=dict(size=12, color="#333"),
                    row=2, 
                    col=1,
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.2)',
                    zeroline=False,
                    fixedrange=True  # Prevents y-axis zooming for cleaner interaction
                )
                
                # Add different background for the indicator subplot
                fig.add_shape(
                    type="rect",
                    x0=0,
                    x1=1,
                    y0=0,
                    y1=0.28,
                    fillcolor="rgba(240, 245, 250, 0.8)",
                    line=dict(width=0),
                    xref="paper",
                    yref="paper",
                    layer="below"
                )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Chickpeas Spot Price with Technical Indicators',
                    'y':0.98,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24, color='#2c3e50', family='Arial, sans-serif')
                },
                xaxis_title={
                    'text': 'Date',
                    'font': dict(size=14, color='#2c3e50', family='Arial, sans-serif'),
                    'standoff': 15
                },
                yaxis_title={
                    'text': 'Price',
                    'font': dict(size=14, color='#2c3e50', family='Arial, sans-serif'),
                    'standoff': 15
                },
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color="#333"
                ),
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,  # Move legend down to create more space
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, family='Arial, sans-serif'),
                    bordercolor="#DDDDDD",
                    borderwidth=1,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    itemsizing='constant'
                ),
                margin=dict(l=80, r=50, t=120, b=140),
                height=900,  # Increased height for better visualization
                plot_bgcolor='rgba(248, 249, 250, 1)',
                paper_bgcolor='white',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=14,
                    font_family="Arial, sans-serif",
                    bordercolor='rgba(0, 0, 0, 0.1)'
                ),
                modebar=dict(
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    color='#2c3e50',
                    activecolor='#3498db'
                )
            )
            
            # Add a visual separator line between main chart and indicator
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=0.285,  # Position just above the indicator section
                y1=0.285,
                line=dict(
                    color="rgba(0, 0, 0, 0.2)",
                    width=1.5,
                    dash="solid"
                ),
                xref="paper",
                yref="paper"
            )
            
            # Add a background band for the space between charts
            fig.add_shape(
                type="rect",
                x0=0,
                x1=1,
                y0=0.28,
                y1=0.30,
                fillcolor="rgba(230, 236, 245, 0.7)",
                line=dict(width=0),
                xref="paper",
                yref="paper",
                layer="below"
            )
            
            # Add a label for the date axis with arrow
            fig.add_annotation(
                x=0.5,
                y=0.015,
                xref="paper",
                yref="paper",
                text="Date",
                showarrow=False,
                font=dict(
                    family="Arial, sans-serif",
                    size=13,
                    color="#2c3e50"
                ),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.1)",
                borderwidth=1,
                borderpad=4
            )
            
            # Improve the indicator subplot title
            fig.update_annotations(
                selector=dict(text="Indicators"),
                font=dict(size=14, color="#2c3e50", family="Arial, sans-serif"),
                y=0.265  # Adjusted position 
            )
            
            # Add candlestick-specific layout settings
            if chart_type == 'candlestick':
                fig.update_layout(
                    xaxis_rangeslider_visible=False  # Hide rangeslider for cleaner look
                )
            
            # Update axis layouts
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.2)',
                zeroline=False,
                tickangle=0,
                tickfont=dict(size=12, family='Arial, sans-serif'),
                tickformat='%b %Y',
                nticks=10,
                showspikes=True,  # Show spikes for better hover experience
                spikemode='across',
                spikesnap='cursor',
                spikecolor='rgba(0, 0, 0, 0.3)',
                spikedash='solid',
                spikethickness=1
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.2)',
                zeroline=False,
                showticklabels=True,
                tickfont=dict(size=12, family='Arial, sans-serif'),
                tickformat=',d',
                showspikes=True,  # Show spikes for better hover experience
                spikemode='across',
                spikesnap='cursor',
                spikecolor='rgba(0, 0, 0, 0.3)',
                spikedash='solid',
                spikethickness=1,
                row=1, col=1
            )
            
            # Improve x-axis for the indicator panel
            fig.update_xaxes(
                row=2,
                col=1,
                title_text="Date",
                title_font=dict(size=12, color="#2c3e50"),
                title_standoff=15,
                showticklabels=True,
                tickfont=dict(size=11),
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.2)'
            )
        
        else:
            # Create a single plot for just the price data
            fig = go.Figure()
            
            # Add main price trace - either line or candlestick
            if chart_type == 'candlestick':
                fig.add_trace(go.Candlestick(
                    x=df_indicators[date_col],
                    open=df_indicators['open'],
                    high=df_indicators['high'],
                    low=df_indicators['low'],
                    close=df_indicators[price_col],
                    name='OHLC'
                ))
                print("Added basic candlestick trace")
            else:
                # Default to line chart
                fig.add_trace(go.Scatter(
                    x=df_indicators[date_col], 
                    y=df_indicators[price_col],
                    mode='lines',
                    name='Price',
                    line=dict(color='#000000', width=2.5)
                ))
            
            # Add SMA 20
            fig.add_trace(go.Scatter(
                x=df_indicators[date_col], 
                y=df_indicators['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#1f77b4', width=1.5, dash='solid')
            ))
            
            # Add SMA 50
            fig.add_trace(go.Scatter(
                x=df_indicators[date_col], 
                y=df_indicators['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='#d62728', width=1.5, dash='solid')
            ))
            
            # Add Bollinger Bands
            fig.add_trace(go.Scatter(
                x=df_indicators[date_col], 
                y=df_indicators['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(0, 128, 0, 0.7)', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_indicators[date_col], 
                y=df_indicators['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(0, 128, 0, 0.7)', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title=f'Chickpeas Spot Price',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,  # Hide the rangeslider for cleaner look
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template='plotly_white',
                showlegend=True,
                height=600,
                hovermode='x unified',
                margin=dict(l=60, r=40, t=80, b=50)
            )
        
        # Convert to JSON for sending to client
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(data=json.loads(graphJSON)['data'], layout=json.loads(graphJSON)['layout'])
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable for production or default to 8051
    port = int(os.environ.get('PORT', 8051))
    app.run(host='0.0.0.0', port=port)

# For Vercel deployment
app.debug = False 