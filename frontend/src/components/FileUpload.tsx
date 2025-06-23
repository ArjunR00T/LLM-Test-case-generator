import React, { useState, useRef } from 'react';
import { Upload, File, X, Download, CheckCircle, AlertCircle, Loader2, FolderOpen, BarChart3 } from 'lucide-react';

interface UploadedFile {
  name: string;
  size: number;
  type: string;
  content?: string;
}

interface ProcessedResult {
  fileName: string;
  userStoriesCount: number;
  testCasesGenerated: number;
  status: 'processing' | 'completed' | 'error';
  downloadUrl?: string;
  processingTime?: string;
  coverage?: number;
}

const FileUpload: React.FC = () => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedResults, setProcessedResults] = useState<ProcessedResult[]>([]);
  // const [selectedProject, setSelectedProject] = useState('ecommerce-platform');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // const projects = [
  //   { id: 'ecommerce-platform', name: 'E-Commerce Platform' },
  //   { id: 'user-management', name: 'User Management System' },
  //   { id: 'payment-gateway', name: 'Payment Gateway' },
  //   { id: 'inventory-system', name: 'Inventory Management' }
  // ];

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    handleFiles(files);
  };

  const handleFiles = (files: File[]) => {
    const validFiles = files.filter(file => 
      file.type === 'text/plain' || 
      file.type === 'text/csv' || 
      file.type === 'application/vnd.ms-excel' ||
      file.name.endsWith('.txt') || 
      file.name.endsWith('.csv') ||
      file.name.endsWith('.xlsx')
    );

    validFiles.forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const newFile: UploadedFile = {
          name: file.name,
          size: file.size,
          type: file.type,
          content: e.target?.result as string
        };
        setUploadedFiles(prev => [...prev, newFile]);
      };
      reader.readAsText(file);
    });
  };

  const removeFile = (fileName: string) => {
    setUploadedFiles(prev => prev.filter(file => file.name !== fileName));
    setProcessedResults(prev => prev.filter(result => result.fileName !== fileName));
  };

  const processFiles = async () => {
    if (uploadedFiles.length === 0) return;

    setIsProcessing(true);
    setProcessedResults([]);

    for (const file of uploadedFiles) {
      // Add initial processing state
      setProcessedResults(prev => [...prev, {
        fileName: file.name,
        userStoriesCount: 0,
        testCasesGenerated: 0,
        status: 'processing'
      }]);

      // Simulate processing with realistic timing
      const processingTime = 3000 + Math.random() * 4000;
      await new Promise(resolve => setTimeout(resolve, processingTime));

      // Mock processing results with more realistic data
      const userStoriesCount = Math.floor(Math.random() * 25) + 8;
      const testCasesGenerated = userStoriesCount * (Math.floor(Math.random() * 3) + 3);
      const coverage = Math.floor(Math.random() * 20) + 80;
      const processingTimeFormatted = `${(processingTime / 1000).toFixed(1)}s`;
      
      setProcessedResults(prev => prev.map(result => 
        result.fileName === file.name 
          ? {
              ...result,
              userStoriesCount,
              testCasesGenerated,
              status: Math.random() > 0.05 ? 'completed' : 'error',
              downloadUrl: Math.random() > 0.05 ? '#' : undefined,
              processingTime: processingTimeFormatted,
              coverage
            }
          : result
      ));
    }

    setIsProcessing(false);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'processing':
        return <Loader2 className="h-5 w-5 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      default:
        return null;
    }
  };

  const getFileIcon = (fileName: string) => {
    if (fileName.endsWith('.csv') || fileName.endsWith('.xlsx')) {
      return <BarChart3 className="h-8 w-8 text-green-600" />;
    }
    return <File className="h-8 w-8 text-blue-600" />;
  };

  const totalStats = processedResults.reduce((acc, result) => {
    if (result.status === 'completed') {
      acc.userStories += result.userStoriesCount;
      acc.testCases += result.testCasesGenerated;
      acc.completed += 1;
    }
    return acc;
  }, { userStories: 0, testCases: 0, completed: 0 });

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center space-x-3 mb-2">
          <Upload className="h-8 w-8 text-blue-600" />
          <h1 className="text-3xl font-bold text-gray-900">Bulk File Processing</h1>
        </div>
        <p className="text-gray-600">Upload and process multiple files containing user stories for batch test case generation</p>
      </div>

      {/* Project Selection */}
      {/* <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
        <label htmlFor="project" className="block text-sm font-semibold text-gray-700 mb-2">
          Target Project
        </label>
        <select
          id="project"
          value={selectedProject}
          onChange={(e) => setSelectedProject(e.target.value)}
          className="w-full max-w-md px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white text-gray-900"
        >
          {projects.map((project) => (
            <option key={project.id} value={project.id}>
              {project.name}
            </option>
          ))}
        </select>
      </div> */}

      {/* File Upload Area */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
            dragActive 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <FolderOpen className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <p className="text-lg font-medium text-gray-700 mb-2">
            Drop your files here, or{' '}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="text-blue-600 hover:text-blue-700 underline font-semibold"
            >
              browse files
            </button>
          </p>
          <p className="text-sm text-gray-500 mb-4">
            Supports TXT, CSV, and Excel files up to 10MB each
          </p>
          <div className="flex items-center justify-center space-x-4 text-xs text-gray-400">
            <span className="flex items-center space-x-1">
              <File className="h-4 w-4" />
              <span>TXT</span>
            </span>
            <span className="flex items-center space-x-1">
              <BarChart3 className="h-4 w-4" />
              <span>CSV</span>
            </span>
            <span className="flex items-center space-x-1">
              <BarChart3 className="h-4 w-4" />
              <span>XLSX</span>
            </span>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".txt,.csv,.xlsx,text/plain,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            onChange={handleFileInput}
            className="hidden"
          />
        </div>

        {/* Uploaded Files List */}
        {uploadedFiles.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Uploaded Files ({uploadedFiles.length})</h3>
            <div className="space-y-3">
              {uploadedFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="flex items-center space-x-3">
                    {getFileIcon(file.name)}
                    <div>
                      <p className="font-medium text-gray-900">{file.name}</p>
                      <p className="text-sm text-gray-500">{formatFileSize(file.size)} • {file.type || 'Unknown type'}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => removeFile(file.name)}
                    className="p-2 hover:bg-red-100 rounded-full transition-colors duration-200"
                    title="Remove file"
                  >
                    <X className="h-4 w-4 text-red-500" />
                  </button>
                </div>
              ))}
            </div>

            <button
              onClick={processFiles}
              disabled={isProcessing || uploadedFiles.length === 0}
              className="mt-6 w-full bg-blue-600 text-white font-semibold py-3 px-6 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 flex items-center justify-center space-x-2"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Processing Files...</span>
                </>
              ) : (
                <>
                  <Upload className="h-5 w-5" />
                  <span>Process All Files</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Processing Summary */}
      {processedResults.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total User Stories</p>
                <p className="text-2xl font-bold text-blue-600">{totalStats.userStories}</p>
              </div>
              <File className="h-8 w-8 text-blue-600" />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Test Cases Generated</p>
                <p className="text-2xl font-bold text-green-600">{totalStats.testCases}</p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-600" />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Files Completed</p>
                <p className="text-2xl font-bold text-purple-600">{totalStats.completed}/{processedResults.length}</p>
              </div>
              <BarChart3 className="h-8 w-8 text-purple-600" />
            </div>
          </div>
        </div>
      )}

      {/* Processing Results */}
      {processedResults.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-6">Processing Results</h2>
          
          <div className="space-y-4">
            {processedResults.map((result, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-6 bg-gray-50">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(result.status)}
                    <div>
                      <h3 className="font-semibold text-gray-900">{result.fileName}</h3>
                      {result.processingTime && (
                        <p className="text-sm text-gray-500">Processed in {result.processingTime}</p>
                      )}
                    </div>
                  </div>
                  
                  {result.status === 'completed' && result.downloadUrl && (
                    <div className="flex items-center space-x-2">
                      <button className="flex items-center space-x-2 px-4 py-2 bg-green-100 text-green-700 rounded-md hover:bg-green-200 transition-colors duration-200">
                        <Download className="h-4 w-4" />
                        <span>Download</span>
                      </button>
                    </div>
                  )}
                </div>

                {result.status === 'processing' && (
                  <div className="space-y-2">
                    <p className="text-gray-600">Analyzing user stories and generating test cases...</p>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{ width: '45%' }}></div>
                    </div>
                  </div>
                )}

                {result.status === 'completed' && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                      <p className="text-blue-700 font-medium text-sm">User Stories</p>
                      <p className="text-2xl font-bold text-blue-800">{result.userStoriesCount}</p>
                    </div>
                    <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                      <p className="text-green-700 font-medium text-sm">Test Cases</p>
                      <p className="text-2xl font-bold text-green-800">{result.testCasesGenerated}</p>
                    </div>
                    <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                      <p className="text-purple-700 font-medium text-sm">Coverage</p>
                      <p className="text-2xl font-bold text-purple-800">{result.coverage}%</p>
                    </div>
                    <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                      <p className="text-orange-700 font-medium text-sm">Avg per Story</p>
                      <p className="text-2xl font-bold text-orange-800">{Math.round(result.testCasesGenerated / result.userStoriesCount)}</p>
                    </div>
                  </div>
                )}

                {result.status === 'error' && (
                  <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                    <p className="text-red-700 font-medium">Processing Error</p>
                    <p className="text-red-600 text-sm mt-1">
                      Unable to process file. Please verify the file format and content structure, then try again.
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;