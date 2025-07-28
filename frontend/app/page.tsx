"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { motion, useAnimation } from "framer-motion"
import {
  Book,
  Ruler,
  Briefcase,
  Upload,
  Pencil,
  Globe,
  Clipboard,
  FileText,
  AlertCircle,
  Loader2,
  CheckCircle2,
  XCircle,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Chart as ChartJS, ArcElement, Tooltip, Legend, BarElement, CategoryScale, LinearScale } from "chart.js"
import { Pie, Bar } from "react-chartjs-2"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { toast } from "sonner"

ChartJS.register(ArcElement, Tooltip, Legend, BarElement, CategoryScale, LinearScale)

type AnalysisResult = {
  ai_percentage: number
  confidence: number
  individual_predictions: {
    custom: number
    transformer: number
    linguistic: number
  }
  linguistic_features?: Record<string, number>
  processing_time?: number
  text_length?: number
  cached?: boolean
  status?: "accepted" | "rejected"
  error?: string
}

type BatchResult = {
  filename: string
  ai_percentage: number
  confidence: number
  individual_predictions: {
    custom: number
    transformer: number
    linguistic: number
  }
  status?: "accepted" | "rejected" | "error"
  error?: string
}

const FloatingIcon = ({
  icon: Icon,
  x,
  y,
  containerWidth,
  containerHeight,
}: {
  icon: React.ElementType
  x: number
  y: number
  containerWidth: number
  containerHeight: number
}) => {
  const controls = useAnimation()

  useEffect(() => {
    controls.start({
      y: [y, y + 20, y],
      transition: {
        duration: 4,
        repeat: Number.POSITIVE_INFINITY,
        repeatType: "reverse",
        ease: "easeInOut",
      },
    })
  }, [controls, y])

  const normalizedX = (x / 100) * containerWidth
  const normalizedY = (y / 100) * containerHeight

  return (
    <motion.div
      className="absolute text-blue-400 opacity-30"
      style={{ left: `${normalizedX}px`, top: `${normalizedY}px` }}
      animate={controls}
      whileHover={{ scale: 1.3, opacity: 0.9 }}
    >
      <Icon size={64} />
    </motion.div>
  )
}

export default function AssignmentSubmission() {
  const [file, setFile] = useState<File | null>(null)
  const [batchFiles, setBatchFiles] = useState<File[]>([])
  const [textInput, setTextInput] = useState("")
  const [activeTab, setActiveTab] = useState<"single" | "batch" | "text">("single")
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [batchResults, setBatchResults] = useState<BatchResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [rateLimited, setRateLimited] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const [dimensions, setDimensions] = useState({ width: 1000, height: 800 })

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0])
    }
  }

  const handleBatchFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setBatchFiles(Array.from(event.target.files))
    }
  }

  // Simulate API call for demo purposes
  const simulateAnalysis = async (): Promise<AnalysisResult> => {
    await new Promise((resolve) => setTimeout(resolve, 2000))

    // Simulate rate limiting (10% chance)
    if (Math.random() < 0.1) {
      throw new Error("Rate limit exceeded")
    }

    const aiPercentage = Math.random() * 100
    return {
      ai_percentage: aiPercentage,
      confidence: 0.75 + Math.random() * 0.2,
      individual_predictions: {
        custom: aiPercentage + (Math.random() - 0.5) * 20,
        transformer: aiPercentage + (Math.random() - 0.5) * 15,
        linguistic: aiPercentage + (Math.random() - 0.5) * 25,
      },
      linguistic_features: {
        word_count: Math.floor(Math.random() * 500) + 100,
        sentence_count: Math.floor(Math.random() * 30) + 5,
        avg_sentence_length: 15 + Math.random() * 10,
        avg_word_length: 4 + Math.random() * 2,
        flesch_reading_ease: Math.random() * 100,
        flesch_kincaid_grade: Math.random() * 20,
        NN_ratio: Math.random() * 0.5,
        VB_ratio: Math.random() * 0.3,
        JJ_ratio: Math.random() * 0.2,
        RB_ratio: Math.random() * 0.15,
        punctuation_ratio: Math.random() * 0.1,
        vocabulary_diversity: Math.random(),
        complex_sentences: Math.random() * 0.4,
        repetition_score: Math.random() * 0.3,
      },
      processing_time: Math.random() * 2 + 0.5,
      cached: Math.random() < 0.3,
      status: aiPercentage > 60 ? "rejected" : "accepted",
    }
  }

  const simulateBatchAnalysis = async (): Promise<BatchResult[]> => {
    await new Promise((resolve) => setTimeout(resolve, 3000))

    return batchFiles.map((file) => {
      const aiPercentage = Math.random() * 100
      return {
        filename: file.name,
        ai_percentage: aiPercentage,
        confidence: 0.75 + Math.random() * 0.2,
        individual_predictions: {
          custom: aiPercentage + (Math.random() - 0.5) * 20,
          transformer: aiPercentage + (Math.random() - 0.5) * 15,
          linguistic: aiPercentage + (Math.random() - 0.5) * 25,
        },
        status: aiPercentage > 60 ? "rejected" : "accepted",
      }
    })
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setIsLoading(true)
    setResult(null)
    setBatchResults([])
    setRateLimited(false)

    try {
      if (activeTab === "batch" && batchFiles.length > 0) {
        const results = await simulateBatchAnalysis()
        setBatchResults(results)
        toast.success(`Processed ${results.length} files`)
      } else {
        const analysisResult = await simulateAnalysis()
        setResult(analysisResult)

        if (analysisResult.status === "accepted") {
          toast.success("Submission accepted!", {
            description: `AI content: ${analysisResult.ai_percentage.toFixed(2)}%`,
          })
        } else if (analysisResult.status === "rejected") {
          toast.error("Submission rejected", {
            description: `High AI content detected: ${analysisResult.ai_percentage.toFixed(2)}%`,
          })
        }
      }
    } catch (error) {
      if (error instanceof Error && error.message === "Rate limit exceeded") {
        setRateLimited(true)
        toast.error("Rate limit exceeded. Please try again later.")
      } else {
        toast.error("Error processing submission", {
          description: error instanceof Error ? error.message : "Unknown error occurred",
        })
      }
    } finally {
      setIsLoading(false)
    }
  }

  const renderPieChart = (aiPercentage: number) => {
    const chartData = {
      labels: ["AI Generated", "Human Generated"],
      datasets: [
        {
          data: [aiPercentage, 100 - aiPercentage],
          backgroundColor: ["rgba(255, 99, 132, 0.6)", "rgba(75, 192, 192, 0.6)"],
          borderColor: ["rgba(255, 99, 132, 1)", "rgba(75, 192, 192, 1)"],
          borderWidth: 1,
        },
      ],
    }

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom" as const,
        },
      },
    }

    return <Pie data={chartData} options={options} className="h-64" />
  }

  const renderModelComparison = (predictions: AnalysisResult["individual_predictions"]) => {
    const data = {
      labels: ["Custom Model", "Transformer", "Linguistic"],
      datasets: [
        {
          label: "AI Probability (%)",
          data: [predictions.custom, predictions.transformer, predictions.linguistic],
          backgroundColor: ["rgba(54, 162, 235, 0.6)", "rgba(255, 159, 64, 0.6)", "rgba(153, 102, 255, 0.6)"],
          borderColor: ["rgba(54, 162, 235, 1)", "rgba(255, 159, 64, 1)", "rgba(153, 102, 255, 1)"],
          borderWidth: 1,
        },
      ],
    }

    const options = {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
        },
      },
    }

    return <Bar data={data} options={options} className="h-64" />
  }

  const renderLinguisticFeatures = (features: Record<string, number>) => {
    const featureGroups = [
      {
        title: "Basic Statistics",
        features: ["word_count", "sentence_count", "avg_sentence_length", "avg_word_length"],
      },
      {
        title: "Readability",
        features: ["flesch_reading_ease", "flesch_kincaid_grade"],
      },
      {
        title: "Part-of-Speech",
        features: ["NN_ratio", "VB_ratio", "JJ_ratio", "RB_ratio"],
      },
      {
        title: "Complexity",
        features: ["punctuation_ratio", "vocabulary_diversity", "complex_sentences", "repetition_score"],
      },
    ]

    return (
      <div className="grid gap-4 md:grid-cols-2">
        {featureGroups.map((group) => (
          <Card key={group.title}>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">{group.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableBody>
                  {group.features.map((key) => (
                    <TableRow key={key}>
                      <TableCell className="font-medium capitalize">{key.replace(/_/g, " ")}</TableCell>
                      <TableCell className="text-right">{features[key]?.toFixed(3) || "N/A"}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  const renderBatchResults = () => {
    if (batchResults.length === 0) return null

    return (
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Batch Results</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Filename</TableHead>
                <TableHead>AI %</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {batchResults.map((result, index) => (
                <TableRow key={index}>
                  <TableCell className="font-medium max-w-[200px] truncate">{result.filename}</TableCell>
                  <TableCell>
                    <Progress value={result.ai_percentage} className="h-2" />
                    <span className="text-sm ml-2">{result.ai_percentage.toFixed(1)}%</span>
                  </TableCell>
                  <TableCell>{result.confidence.toFixed(3)}</TableCell>
                  <TableCell>
                    {result.status === "accepted" ? (
                      <Badge variant="default" className="bg-green-500">
                        <CheckCircle2 className="h-4 w-4 mr-1" /> Accepted
                      </Badge>
                    ) : result.status === "rejected" ? (
                      <Badge variant="destructive">
                        <XCircle className="h-4 w-4 mr-1" /> Rejected
                      </Badge>
                    ) : (
                      <Badge variant="outline">
                        <AlertCircle className="h-4 w-4 mr-1" /> Error
                      </Badge>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    )
  }

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        })
      }
    }

    updateDimensions()
    window.addEventListener("resize", updateDimensions)

    return () => {
      window.removeEventListener("resize", updateDimensions)
    }
  }, [])

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (containerRef.current) {
        const { left, top } = containerRef.current.getBoundingClientRect()
        setMousePosition({
          x: event.clientX - left,
          y: event.clientY - top,
        })
      }
    }

    window.addEventListener("mousemove", handleMouseMove)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
    }
  }, [])

  return (
    <div
      ref={containerRef}
      className="min-h-screen bg-blue-50 flex flex-col items-center justify-center p-4 relative overflow-hidden"
    >
      <FloatingIcon
        icon={Briefcase}
        x={10}
        y={20}
        containerWidth={dimensions.width}
        containerHeight={dimensions.height}
      />
      <FloatingIcon icon={Ruler} x={90} y={15} containerWidth={dimensions.width} containerHeight={dimensions.height} />
      <FloatingIcon icon={Book} x={50} y={80} containerWidth={dimensions.width} containerHeight={dimensions.height} />
      <FloatingIcon icon={Pencil} x={15} y={50} containerWidth={dimensions.width} containerHeight={dimensions.height} />
      <FloatingIcon icon={Globe} x={80} y={10} containerWidth={dimensions.width} containerHeight={dimensions.height} />
      <FloatingIcon
        icon={Clipboard}
        x={30}
        y={60}
        containerWidth={dimensions.width}
        containerHeight={dimensions.height}
      />

      <motion.div
        className="absolute w-48 h-48 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl opacity-70"
        animate={{
          x: mousePosition.x - 20,
          y: mousePosition.y - 20,
        }}
        transition={{ type: "spring", damping: 10, stiffness: 50 }}
      />

      <header className="text-center mb-8 relative z-10">
        <h1 className="text-3xl md:text-4xl font-bold text-blue-800 mb-2">AI Content Detection System</h1>
        <p className="text-lg md:text-xl text-blue-600">Analyze documents for AI-generated content</p>
      </header>

      <Card className="w-full max-w-2xl relative z-10">
        <CardHeader>
          <CardTitle className="text-xl md:text-2xl text-center text-blue-800">
            {activeTab === "single"
              ? "Single File Analysis"
              : activeTab === "batch"
                ? "Batch Analysis"
                : "Text Analysis"}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as any)} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="single">Single File</TabsTrigger>
              <TabsTrigger value="batch">Batch Files</TabsTrigger>
              <TabsTrigger value="text">Text Input</TabsTrigger>
            </TabsList>

            <TabsContent value="single">
              <form onSubmit={handleSubmit} className="space-y-4">
                <Input
                  type="file"
                  accept=".pdf,.docx,.txt"
                  onChange={handleFileChange}
                  className="hidden"
                  id="fileInput"
                />
                <label htmlFor="fileInput" className="cursor-pointer">
                  <Button variant="outline" className="w-full hover:bg-blue-500 hover:text-white" asChild>
                    <div>
                      {file ? (
                        <>
                          <FileText className="mr-2 h-4 w-4" /> {file.name}
                        </>
                      ) : (
                        <>
                          <Upload className="mr-2 h-4 w-4" /> Select File (PDF, DOCX, TXT)
                        </>
                      )}
                    </div>
                  </Button>
                </label>
                <Button
                  type="submit"
                  className="w-full hover:bg-green-500 hover:text-white"
                  disabled={!file || isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze File"
                  )}
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="batch">
              <form onSubmit={handleSubmit} className="space-y-4">
                <Input
                  type="file"
                  accept=".pdf,.docx,.txt"
                  onChange={handleBatchFileChange}
                  className="hidden"
                  id="batchFileInput"
                  multiple
                />
                <label htmlFor="batchFileInput" className="cursor-pointer">
                  <Button variant="outline" className="w-full hover:bg-blue-500 hover:text-white" asChild>
                    <div>
                      {batchFiles.length > 0 ? (
                        <>
                          <FileText className="mr-2 h-4 w-4" /> {batchFiles.length} files selected
                        </>
                      ) : (
                        <>
                          <Upload className="mr-2 h-4 w-4" /> Select Multiple Files
                        </>
                      )}
                    </div>
                  </Button>
                </label>
                <Button
                  type="submit"
                  className="w-full hover:bg-green-500 hover:text-white"
                  disabled={batchFiles.length === 0 || isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing {batchFiles.length} files...
                    </>
                  ) : (
                    "Analyze Batch"
                  )}
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="text">
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid w-full gap-1.5">
                  <textarea
                    className="min-h-[120px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                    placeholder="Paste your text here for analysis..."
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                  />
                  <p className="text-sm text-muted-foreground">{textInput.length} characters</p>
                </div>
                <Button
                  type="submit"
                  className="w-full hover:bg-green-500 hover:text-white"
                  disabled={!textInput.trim() || isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze Text"
                  )}
                </Button>
              </form>
            </TabsContent>
          </Tabs>

          {rateLimited && (
            <div className="mt-4 p-4 bg-red-100 border border-red-200 rounded-md text-red-800">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 mr-2" />
                <span>Rate limit exceeded. Please try again later.</span>
              </div>
            </div>
          )}

          {result && (
            <div className="mt-6 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Analysis Results</span>
                    <Badge
                      variant={result.status === "accepted" ? "default" : "destructive"}
                      className={result.status === "accepted" ? "bg-green-500" : ""}
                    >
                      {result.status === "accepted" ? "Accepted" : "Rejected"}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">AI Content Probability</p>
                      <p className="text-3xl font-bold">{result.ai_percentage.toFixed(2)}%</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Confidence</p>
                      <p className="text-3xl font-bold">{(result.confidence * 100).toFixed(2)}%</p>
                    </div>
                  </div>

                  <Progress value={result.ai_percentage} className="h-2" />

                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Content Breakdown</h3>
                      {renderPieChart(result.ai_percentage)}
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Model Comparison</h3>
                      {renderModelComparison(result.individual_predictions)}
                    </div>
                  </div>

                  {result.linguistic_features && (
                    <>
                      <h3 className="text-lg font-semibold">Linguistic Features</h3>
                      {renderLinguisticFeatures(result.linguistic_features)}
                    </>
                  )}
                </CardContent>
                <CardFooter className="text-sm text-muted-foreground">
                  <p>
                    Processed in {result.processing_time?.toFixed(3) || "N/A"} seconds •
                    {result.cached ? " Cached result" : " Fresh analysis"}
                  </p>
                </CardFooter>
              </Card>
            </div>
          )}
        </CardContent>
      </Card>

      {activeTab === "batch" && renderBatchResults()}
    </div>
  )
}
