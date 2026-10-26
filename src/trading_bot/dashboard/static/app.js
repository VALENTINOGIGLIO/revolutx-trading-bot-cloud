// Trading Bot Web Dashboard - Lightweight Charts & SSE Client

document.addEventListener("DOMContentLoaded", () => {
    const chartContainer = document.getElementById("chart");
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");
    const symbolBadge = document.getElementById("symbol-badge");

    // KPI card elements
    const kpiEquity = document.getElementById("kpi-equity");
    const kpiCash = document.getElementById("kpi-cash");
    const kpiTrades = document.getElementById("kpi-trades");
    const kpiStatus = document.getElementById("kpi-status");

    const tradesTableBody = document.getElementById("trades-table-body");

    let chart = null;
    let candleSeries = null;
    let markersList = [];

    // 1. Initialize Lightweight Charts
    function initChart() {
        if (typeof LightweightCharts === "undefined") {
            console.error("LightweightCharts library not loaded.");
            chartContainer.innerHTML = "<div style='padding:20px;text-align:center;color:#ef4444;'>Error: Charting library failed to load.</div>";
            return;
        }

        chart = LightweightCharts.createChart(chartContainer, {
            width: chartContainer.clientWidth,
            height: chartContainer.clientHeight || 480,
            layout: {
                background: { color: "#111827" },
                textColor: "#9ca3af",
            },
            grid: {
                vertLines: { color: "#1f2937" },
                horzLines: { color: "#1f2937" },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
            },
            rightPriceScale: {
                borderColor: "#1f2937",
            },
            timeScale: {
                borderColor: "#1f2937",
                timeVisible: true,
                secondsVisible: false,
            },
        });

        candleSeries = chart.addCandlestickSeries({
            upColor: "#10b981",
            downColor: "#ef4444",
            borderUpColor: "#10b981",
            borderDownColor: "#ef4444",
            wickUpColor: "#10b981",
            wickDownColor: "#ef4444",
        });

        window.addEventListener("resize", () => {
            if (chart && chartContainer) {
                chart.applyOptions({ width: chartContainer.clientWidth });
            }
        });
    }

    // 2. Fetch and populate Market Data
    async function loadMarketData() {
        try {
            let resp = await fetch("/api/candles");
            if (!resp.ok) {
                resp = await fetch("/api/market-data");
            }
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            if (Array.isArray(data) && data.length > 0 && candleSeries) {
                // Ensure candles sorted by time
                data.sort((a, b) => a.time - b.time);
                candleSeries.setData(data);
                chart.timeScale().fitContent();
            }
        } catch (err) {
            console.warn("Failed to load initial market data:", err);
        }
    }

    // 3. Fetch and populate Trades & Markers
    async function loadTrades() {
        try {
            const resp = await fetch("/api/trades");
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            if (Array.isArray(data)) {
                markersList = data;
                // Sort markers by time ascending for Lightweight Charts
                markersList.sort((a, b) => a.time - b.time);
                if (candleSeries) {
                    candleSeries.setMarkers(markersList);
                }
                renderTradesTable(markersList);
            }
        } catch (err) {
            console.warn("Failed to load trade markers:", err);
        }
    }

    // 4. Fetch System Status & Balances
    async function loadStatus() {
        try {
            const resp = await fetch("/api/status");
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            updateStatusUI(data);
        } catch (err) {
            console.warn("Failed to load bot status:", err);
        }
    }

    function updateStatusUI(data) {
        if (!data) return;
        if (data.symbol && symbolBadge) {
            symbolBadge.textContent = data.symbol;
        }
        if (kpiEquity && data.portfolio_balance !== undefined) {
            kpiEquity.textContent = `${Number(data.portfolio_balance).toFixed(2)} €`;
            const kpiNetProfit = document.getElementById("kpi-net-profit");
            if (kpiNetProfit) {
                const netProfit = data.net_profit !== undefined ? Number(data.net_profit) : 0.0;
                const sign = netProfit >= 0 ? "+" : "";
                kpiNetProfit.textContent = `Guadagni Netti: ${sign}${netProfit.toFixed(2)} €`;
                kpiNetProfit.style.color = netProfit >= 0 ? "#10b981" : "#ef4444";
            }
        }
        if (kpiCash && data.cash_balance !== undefined) {
            kpiCash.textContent = `${Number(data.cash_balance).toFixed(2)} €`;
        }
        if (kpiTrades && data.total_trades !== undefined) {
            kpiTrades.textContent = data.total_trades;
        }
        if (kpiStatus && data.status) {
            kpiStatus.textContent = data.status.toUpperCase();
            kpiStatus.style.color = data.status.toLowerCase() === "running" ? "#10b981" : "#9ca3af";
        }
    }

    function renderTradesTable(markers) {
        if (!tradesTableBody) return;
        tradesTableBody.innerHTML = "";

        // Display in reverse chronological order
        const sorted = [...markers].sort((a, b) => b.time - a.time);

        if (sorted.length === 0) {
            tradesTableBody.innerHTML = "<tr><td colspan='5' style='text-align:center;color:#6b7280;padding:1rem;'>No trades executed yet</td></tr>";
            return;
        }

        sorted.forEach((item) => {
            const row = document.createElement("tr");
            const dt = new Date(item.time * 1000);
            const timeStr = dt.toISOString().replace("T", " ").substring(0, 19);

            const isBuy = (item.action || (item.shape === "arrowUp" ? "BUY" : "SELL")) === "BUY";
            const badgeClass = isBuy ? "buy" : "sell";
            const actionText = isBuy ? "BUY" : "SELL";

            const priceStr = item.price ? Number(item.price).toFixed(2) : "-";
            const sizeStr = item.size ? Number(item.size).toFixed(6) : "-";

            let pnlHtml = "<span class='text-muted'>-</span>";
            if (!isBuy && item.pnl !== undefined) {
                const pnlNum = Number(item.pnl);
                const pnlClass = pnlNum >= 0 ? "text-positive" : "text-negative";
                const pnlSign = pnlNum >= 0 ? "+" : "";
                pnlHtml = `<span class="${pnlClass}">${pnlSign}${pnlNum.toFixed(2)} €</span>`;
            }

            row.innerHTML = `
                <td>${timeStr}</td>
                <td><span class="badge ${badgeClass}">${actionText}</span></td>
                <td>${priceStr}</td>
                <td>${sizeStr}</td>
                <td>${pnlHtml}</td>
            `;
            tradesTableBody.appendChild(row);
        });
    }

    // 5. Connect to SSE Live Stream
    function initSSE() {
        const evtSource = new EventSource("/api/stream");

        evtSource.onopen = () => {
            if (statusDot) statusDot.classList.remove("disconnected");
            if (statusText) statusText.textContent = "CONNECTED (SSE)";
        };

        evtSource.addEventListener("candle", (e) => {
            try {
                const candle = JSON.parse(e.data);
                if (candle && candleSeries) {
                    candleSeries.update(candle);
                }
            } catch (err) {
                console.error("Error handling candle SSE:", err);
            }
        });

        evtSource.addEventListener("trade", (e) => {
            try {
                const tradeData = JSON.parse(e.data);
                if (tradeData) {
                    markersList.push(tradeData);
                    markersList.sort((a, b) => a.time - b.time);
                    if (candleSeries) {
                        candleSeries.setMarkers(markersList);
                    }
                    renderTradesTable(markersList);
                    loadStatus();
                }
            } catch (err) {
                console.error("Error handling trade SSE:", err);
            }
        });

        evtSource.addEventListener("status", (e) => {
            try {
                const statusData = JSON.parse(e.data);
                updateStatusUI(statusData);
            } catch (err) {
                console.error("Error handling status SSE:", err);
            }
        });

        evtSource.onerror = () => {
            if (statusDot) statusDot.classList.add("disconnected");
            if (statusText) statusText.textContent = "DISCONNECTED (RETRYING...)";
        };
    }

    // Initialize UI
    initChart();
    loadMarketData();
    loadTrades();
    loadStatus();
    initSSE();
});
