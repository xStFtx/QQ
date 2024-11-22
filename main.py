import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks
import yfinance as yf
import pywt  # For Wavelet Transform
import plotly.graph_objects as go

# Fetch Data from Yahoo Finance
def fetch_data(tickers, start_date, end_date):
    """Fetch historical stock prices and calculate returns."""
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = np.log(data / data.shift(1)).dropna()  # Log returns
    return returns

# Portfolio Metrics Calculation
def portfolio_metrics(weights, returns, risk_free_rate=0.0):
    """Calculate portfolio return, volatility, and Sharpe ratio."""
    weights = np.array(weights)
    port_return = np.sum(weights * returns.mean()) * 252  # Annualize return (252 trading days)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized volatility
    port_sharpe = (port_return - risk_free_rate) / port_volatility  # Sharpe ratio with risk-free rate
    return port_return, port_volatility, port_sharpe

# Objective Function for Optimization (Volatility or Sharpe Ratio)
def objective_function(weights, returns, risk_free_rate=0.0, maximize_sharpe=False, balance_factor=0.5):
    """Objective function for optimization: Minimize volatility or maximize Sharpe ratio with balancing factor."""
    port_return, port_volatility, port_sharpe = portfolio_metrics(weights, returns, risk_free_rate)
    if maximize_sharpe:
        return -port_sharpe + (1 - balance_factor) * port_volatility  # Multi-objective with a balance factor
    return port_volatility  # Minimize volatility

# Portfolio Optimization with Constraints
def optimize_portfolio(returns, risk_free_rate=0.0, offshore_limit=0.10, maximize_sharpe=False, balance_factor=0.5):
    """Optimize portfolio based on constraints (min volatility or max Sharpe)."""
    num_assets = returns.shape[1]
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        {'type': 'ineq', 'fun': lambda w: offshore_limit - np.sum(w[-3:])}  # Offshore constraint
    ]
    bounds = [(0, 1) for _ in range(num_assets)]  # Weights between 0 and 1
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(
        objective_function, initial_weights, args=(returns, risk_free_rate, maximize_sharpe, balance_factor),
        method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False}
    )
    return result

# Efficient Frontier Calculation
def calculate_efficient_frontier(returns, risk_free_rate=0.0, num_portfolios=10000, offshore_limit=0.10):
    """Generate portfolios for the Efficient Frontier."""
    num_assets = returns.shape[1]
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        if np.sum(weights[-3:]) > offshore_limit:  # Offshore constraint
            continue
        weights_record.append(weights)
        port_return, port_volatility, port_sharpe = portfolio_metrics(weights, returns, risk_free_rate)
        results[0, i] = port_volatility
        results[1, i] = port_return
        results[2, i] = port_sharpe  # Sharpe Ratio

    return results, weights_record

def calculate_wavelet_analysis(returns, weights, plot=True):
    """Perform Wavelet analysis on the portfolio returns."""
    # Calculate portfolio returns using the given weights
    portfolio_returns = np.dot(returns, weights)

    # Perform Continuous Wavelet Transform (CWT)
    coefficients, frequencies = pywt.cwt(portfolio_returns, np.arange(1, 128), 'mexh')
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, len(portfolio_returns), 1, 128], cmap='jet')
        plt.colorbar(label='Magnitude')
        plt.title('Wavelet Transform of Portfolio Returns')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()

    return coefficients, frequencies

def calculate_fourier_analysis(returns, weights, plot=True):
    """Perform Fourier analysis on the portfolio returns."""
    # Calculate portfolio returns using the given weights
    portfolio_returns = np.dot(returns, weights)

    # Apply the Fast Fourier Transform (FFT) to the portfolio returns
    fft_result = np.fft.fft(portfolio_returns)
    
    # Calculate the frequencies corresponding to the FFT result
    fft_freqs = np.fft.fftfreq(len(portfolio_returns), d=1)  # d=1 means daily frequency
    
    # Get the magnitude of the FFT components
    fft_magnitude = np.abs(fft_result)

    # Smooth the FFT results to reduce noise
    smoothed_magnitude = np.convolve(fft_magnitude, np.ones(5)/5, mode='same')

    # Find peaks in the FFT magnitude to identify significant frequencies
    peaks, _ = find_peaks(smoothed_magnitude, height=np.max(smoothed_magnitude)*0.1)

    if plot:
        # Plot the FFT magnitude to show the frequency components
        plt.figure(figsize=(10, 6))
        plt.plot(fft_freqs[:len(fft_freqs)//2], smoothed_magnitude[:len(smoothed_magnitude)//2])  # Only positive frequencies
        plt.scatter(fft_freqs[peaks], smoothed_magnitude[peaks], color='red', label='Significant Peaks')
        plt.title('Fourier Analysis of Portfolio Returns')
        plt.xlabel('Frequency (cycles per day)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()

    return fft_freqs, smoothed_magnitude, peaks

# Visualizing the Efficient Frontier using Plotly for interactivity
def plot_efficient_frontier_interactive(results, optimal_portfolio, labels=("Volatility", "Return")):
    """Visualize the Efficient Frontier with interactive Plotly."""
    max_sharpe_idx = np.argmax(results[2, :])  # Index of maximum Sharpe ratio
    max_sharpe_volatility = results[0, max_sharpe_idx]
    max_sharpe_return = results[1, max_sharpe_idx]

    # Create the plot
    fig = go.Figure()

    # Scatter plot for Efficient Frontier
    fig.add_trace(go.Scatter(
        x=results[0, :], y=results[1, :], mode='markers', marker=dict(color=results[2, :], colorscale='Viridis', size=5),
        name="Efficient Frontier", hovertemplate="Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe Ratio: %{text}",
        text=results[2, :]
    ))

    # Plot Min Volatility Portfolio
    fig.add_trace(go.Scatter(
        x=[optimal_portfolio[0]], y=[optimal_portfolio[1]], mode='markers+text', text=['Min Volatility'],
        marker=dict(color='red', size=10), textposition='top right', name='Min Volatility Portfolio'
    ))

    # Plot Max Sharpe Portfolio
    fig.add_trace(go.Scatter(
        x=[max_sharpe_volatility], y=[max_sharpe_return], mode='markers+text', text=['Max Sharpe'],
        marker=dict(color='blue', size=10), textposition='top right', name='Max Sharpe Portfolio'
    ))

    # Update Layout
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title=labels[0],
        yaxis_title=labels[1],
        template="plotly_dark",
        hovermode="closest",
        showlegend=True
    )

    fig.show()

# Display Results with Allocation
def display_results(optimal_weights, investment_allocation, tickers):
    """Display optimized portfolio weights and investment allocations."""
    formatted_weights = {ticker: f"{weight:.2%}" for ticker, weight in zip(tickers, optimal_weights)}
    formatted_allocation = {ticker: f"R{allocation:.2f}" for ticker, allocation in zip(tickers, investment_allocation)}
    
    print("Optimal Weights (%):", formatted_weights)
    print(f"Expected Return: {optimal_return:.2%}, Expected Volatility: {optimal_volatility:.2%}")
    print("Investment Allocation (ZAR):", formatted_allocation)

# Parameters
tickers = [
    'AGL.JO', 'SOL.JO', 'NPN.JO',  # JSE stocks (local sectors: mining, energy, tech)
    'FSR.JO', 'SHP.JO',            # Financials, retail
    'VOD.JO',                      # Telecoms
    'AAPL', 'MSFT', 'GOOGL'        # Offshore stocks (US tech)
]
start_date = '2020-01-01'
end_date = '2023-01-01'
budget = 100000  # ZAR

# Fetch data
returns = fetch_data(tickers, start_date, end_date)

# Introduce a risk-free rate (e.g., South African 10-year bond yield or US T-bills)
risk_free_rate = 0.05  # 5% risk-free rate

# Optimize portfolio for minimum volatility
optimized_min_vol = optimize_portfolio(returns, risk_free_rate=risk_free_rate, offshore_limit=0.10, maximize_sharpe=False)
optimal_weights = optimized_min_vol.x
optimal_return, optimal_volatility, optimal_sharpe = portfolio_metrics(optimal_weights, returns, risk_free_rate)
investment_allocation = optimal_weights * budget

# Efficient Frontier Calculation
results, weights_record = calculate_efficient_frontier(returns, risk_free_rate=risk_free_rate)

# Plot Efficient Frontier interactively
plot_efficient_frontier_interactive(results, (optimal_volatility, optimal_return))

# Display Results
display_results(optimal_weights, investment_allocation, tickers)

# Fourier and Wavelet Analysis
fft_freqs, fft_magnitude, peaks = calculate_fourier_analysis(returns, optimal_weights)
coefficients, frequencies = calculate_wavelet_analysis(returns, optimal_weights)
