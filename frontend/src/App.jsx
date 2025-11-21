import { useState, useEffect, useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import './App.css'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

function App() {
  const [config, setConfig] = useState({
    model_name: 'MyAttentionModel',
    num_epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
  })
  
  const [isTraining, setIsTraining] = useState(false)
  const [logs, setLogs] = useState([])
  const [chartData, setChartData] = useState({
    labels: [],
    datasets: [
      {
        label: 'Train Loss',
        data: [],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
      {
        label: 'Val Loss',
        data: [],
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      },
    ],
  })
  
  const ws = useRef(null)
  const logsEndRef = useRef(null)

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs])

  const connectWebSocket = () => {
    ws.current = new WebSocket('ws://localhost:8000/ws/train')
    
    ws.current.onopen = () => {
      addLog('Connected to server')
    }
    
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data)
      
      if (message.type === 'progress') {
        const { epoch, train_loss, val_loss } = message.data
        updateChart(epoch, train_loss, val_loss)
        addLog(`Epoch ${epoch}: Train Loss=${train_loss.toFixed(4)}, Val Loss=${val_loss.toFixed(4)}`)
      } else if (message.type === 'finished') {
        setIsTraining(false)
        addLog('Training finished')
      } else if (message.type === 'stopped') {
        setIsTraining(false)
        addLog('Training stopped')
      }
    }
    
    ws.current.onclose = () => {
      addLog('Disconnected from server')
      setIsTraining(false)
    }
  }

  const addLog = (message) => {
    setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`])
  }

  const updateChart = (epoch, trainLoss, valLoss) => {
    setChartData((prev) => ({
      ...prev,
      labels: [...prev.labels, epoch],
      datasets: [
        {
          ...prev.datasets[0],
          data: [...prev.datasets[0].data, trainLoss],
        },
        {
          ...prev.datasets[1],
          data: [...prev.datasets[1].data, valLoss],
        },
      ],
    }))
  }

  const startTraining = () => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      connectWebSocket()
      // Wait for connection
      setTimeout(() => {
        if (ws.current.readyState === WebSocket.OPEN) {
          sendStartCommand()
        } else {
          addLog('Failed to connect to server')
        }
      }, 500)
    } else {
      sendStartCommand()
    }
  }

  const sendStartCommand = () => {
    setChartData({
      labels: [],
      datasets: [
        { ...chartData.datasets[0], data: [] },
        { ...chartData.datasets[1], data: [] },
      ],
    })
    setLogs([])
    setIsTraining(true)
    ws.current.send(JSON.stringify({
      type: 'start',
      config: config
    }))
  }

  const stopTraining = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: 'stop' }))
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Deep Learning Dashboard</h1>
      </header>
      
      <main>
        <div className="panel config-panel">
          <h2>Configuration</h2>
          <div className="form-group">
            <label>Model Name</label>
            <input
              type="text"
              value={config.model_name}
              onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
              disabled={isTraining}
            />
          </div>
          <div className="form-group">
            <label>Epochs</label>
            <input
              type="number"
              value={config.num_epochs}
              onChange={(e) => setConfig({ ...config, num_epochs: parseInt(e.target.value) })}
              disabled={isTraining}
            />
          </div>
          <div className="form-group">
            <label>Batch Size</label>
            <input
              type="number"
              value={config.batch_size}
              onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
              disabled={isTraining}
            />
          </div>
          <div className="form-group">
            <label>Learning Rate</label>
            <input
              type="number"
              step="0.0001"
              value={config.learning_rate}
              onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
              disabled={isTraining}
            />
          </div>
          
          <div className="actions">
            {!isTraining ? (
              <button className="btn-primary" onClick={startTraining}>Start Training</button>
            ) : (
              <button className="btn-danger" onClick={stopTraining}>Stop</button>
            )}
          </div>
        </div>

        <div className="panel chart-panel">
          <h2>Training Progress</h2>
          <div className="chart-container">
            <Line options={{ responsive: true, maintainAspectRatio: false }} data={chartData} />
          </div>
        </div>

        <div className="panel logs-panel">
          <h2>Logs</h2>
          <div className="logs-content">
            {logs.map((log, index) => (
              <div key={index} className="log-entry">{log}</div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
