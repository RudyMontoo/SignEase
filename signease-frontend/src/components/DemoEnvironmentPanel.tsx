/**
 * Demo Environment Panel
 * ======================
 * 
 * Comprehensive demo environment monitoring and control panel
 */

import React, { useState } from 'react'
import { useTheme } from '../hooks/useTheme'
import { useDemoEnvironmentChecker, type DemoEnvironmentCheck } from '../utils/demoEnvironmentChecker'
import Card, { CardHeader, CardTitle, CardContent } from './ui/Card'
import Button from './ui/Button'
import Badge from './ui/Badge'
import DemoScript from './DemoScript'

export interface DemoEnvironmentPanelProps {
  className?: string
}

export const DemoEnvironmentPanel: React.FC<DemoEnvironmentPanelProps> = ({
  className = ''
}) => {
  const { isDark } = useTheme()
  const { report, isChecking, healthStatus, runCheck, exportReport } = useDemoEnvironmentChecker()
  const [activeTab, setActiveTab] = useState<'overview' | 'checks' | 'script' | 'contingency'>('overview')
  const [showOnlyIssues, setShowOnlyIssues] = useState(false)

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass': return '✅'
      case 'warning': return '⚠️'
      case 'fail': return '❌'
      default: return '❓'
    }
  }

  const getOverallStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'success'
      case 'needs_attention': return 'warning'
      case 'not_ready': return 'error'
      default: return 'secondary'
    }
  }

  const getHealthStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'success'
      case 'warning': return 'warning'
      case 'critical': return 'error'
      default: return 'secondary'
    }
  }

  const filteredChecks = showOnlyIssues 
    ? report?.checks.filter(check => check.status !== 'pass') || []
    : report?.checks || []

  const criticalIssues = report?.checks.filter(check => check.critical && check.status === 'fail') || []
  const warnings = report?.checks.filter(check => check.status === 'warning') || []
  const failures = report?.checks.filter(check => check.status === 'fail') || []

  const handleExportReport = () => {
    if (exportReport) {
      const reportData = exportReport()
      const blob = new Blob([reportData], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `demo-environment-report-${new Date().toISOString().slice(0, 19)}.json`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header with Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Demo Environment Status</CardTitle>
            <div className="flex items-center space-x-3">
              {healthStatus && (
                <Badge variant={getHealthStatusColor(healthStatus.status)} size="sm">
                  {healthStatus.status.toUpperCase()}: {healthStatus.message}
                </Badge>
              )}
              {report && (
                <Badge variant={getOverallStatusColor(report.overall)} size="lg">
                  {report.overall.replace('_', ' ').toUpperCase()}
                </Badge>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={runCheck}
                disabled={isChecking}
              >
                {isChecking ? '🔄 Checking...' : '🔍 Run Check'}
              </Button>
            </div>
          </div>
        </CardHeader>

        {report && (
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className={`p-3 rounded-lg ${
                isDark ? 'bg-gray-800' : 'bg-gray-50'
              }`}>
                <div className="text-2xl font-bold text-green-600">
                  {report.checks.filter(c => c.status === 'pass').length}
                </div>
                <div className="text-sm text-gray-600">Checks Passed</div>
              </div>

              <div className={`p-3 rounded-lg ${
                isDark ? 'bg-gray-800' : 'bg-gray-50'
              }`}>
                <div className="text-2xl font-bold text-yellow-600">
                  {warnings.length}
                </div>
                <div className="text-sm text-gray-600">Warnings</div>
              </div>

              <div className={`p-3 rounded-lg ${
                isDark ? 'bg-gray-800' : 'bg-gray-50'
              }`}>
                <div className="text-2xl font-bold text-red-600">
                  {failures.length}
                </div>
                <div className="text-sm text-gray-600">Failures</div>
              </div>

              <div className={`p-3 rounded-lg ${
                isDark ? 'bg-gray-800' : 'bg-gray-50'
              }`}>
                <div className="text-2xl font-bold text-red-800">
                  {criticalIssues.length}
                </div>
                <div className="text-sm text-gray-600">Critical Issues</div>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 border-b border-gray-200 dark:border-gray-700">
        {[
          { id: 'overview', label: 'Overview', icon: '📊' },
          { id: 'checks', label: 'Detailed Checks', icon: '🔍' },
          { id: 'script', label: 'Demo Script', icon: '📝' },
          { id: 'contingency', label: 'Contingency Plans', icon: '🛡️' }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
              activeTab === tab.id
                ? isDark 
                  ? 'bg-gray-800 text-blue-400 border-b-2 border-blue-400'
                  : 'bg-white text-blue-600 border-b-2 border-blue-600'
                : isDark
                  ? 'text-gray-400 hover:text-gray-200'
                  : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <span className="mr-2">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && report && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* System Information */}
          <Card>
            <CardHeader>
              <CardTitle>System Information</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="font-medium">Browser:</span>
                  <span>{report.systemInfo.browser} {report.systemInfo.version}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Platform:</span>
                  <span>{report.systemInfo.platform}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Screen:</span>
                  <span>{report.systemInfo.screenResolution}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">CPU Cores:</span>
                  <span>{report.systemInfo.hardwareConcurrency}</span>
                </div>
                {report.systemInfo.deviceMemory && (
                  <div className="flex justify-between">
                    <span className="font-medium">Device Memory:</span>
                    <span>{report.systemInfo.deviceMemory}GB</span>
                  </div>
                )}
                {report.systemInfo.connectionType && (
                  <div className="flex justify-between">
                    <span className="font-medium">Connection:</span>
                    <span>{report.systemInfo.connectionType}</span>
                  </div>
                )}
                {report.systemInfo.batteryLevel && (
                  <div className="flex justify-between">
                    <span className="font-medium">Battery:</span>
                    <span>{report.systemInfo.batteryLevel}%</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Quick Recommendations */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {report.recommendations.slice(0, 6).map((rec, index) => (
                  <div key={index} className="flex items-start space-x-2">
                    <span className="text-blue-500 mt-1">•</span>
                    <span className="text-sm">{rec}</span>
                  </div>
                ))}
                {report.recommendations.length > 6 && (
                  <div className="text-sm text-gray-500 mt-2">
                    +{report.recommendations.length - 6} more recommendations
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === 'checks' && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Detailed Environment Checks</CardTitle>
              <div className="flex items-center space-x-2">
                <label className="flex items-center space-x-2 text-sm">
                  <input
                    type="checkbox"
                    checked={showOnlyIssues}
                    onChange={(e) => setShowOnlyIssues(e.target.checked)}
                    className="rounded"
                  />
                  <span>Show only issues</span>
                </label>
                <Button variant="outline" size="sm" onClick={handleExportReport}>
                  📄 Export Report
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {filteredChecks.map((check, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg border-l-4 ${
                    check.status === 'pass'
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                      : check.status === 'warning'
                      ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                      : 'border-red-500 bg-red-50 dark:bg-red-900/20'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{getStatusIcon(check.status)}</span>
                      <h3 className="font-medium">{check.name}</h3>
                      {check.critical && (
                        <Badge variant="error" size="sm">Critical</Badge>
                      )}
                    </div>
                    <Badge variant={check.status === 'pass' ? 'success' : check.status === 'warning' ? 'warning' : 'error'} size="sm">
                      {check.status.toUpperCase()}
                    </Badge>
                  </div>
                  <p className="text-sm mb-2">{check.message}</p>
                  {check.recommendation && (
                    <div className={`text-sm p-2 rounded ${
                      isDark ? 'bg-gray-800' : 'bg-white'
                    }`}>
                      <strong>Recommendation:</strong> {check.recommendation}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {activeTab === 'script' && (
        <DemoScript
          onStepComplete={(stepId) => {
            console.log(`Demo step completed: ${stepId}`)
          }}
          onDemoComplete={() => {
            console.log('Demo completed successfully!')
          }}
        />
      )}

      {activeTab === 'contingency' && report && (
        <Card>
          <CardHeader>
            <CardTitle>Contingency Plans</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Critical Issues */}
              {criticalIssues.length > 0 && (
                <div className={`p-4 rounded-lg border-l-4 border-red-500 ${
                  isDark ? 'bg-red-900/20' : 'bg-red-50'
                }`}>
                  <h3 className="font-semibold text-red-600 mb-2">🚨 Critical Issues Detected</h3>
                  <ul className="space-y-1">
                    {criticalIssues.map((issue, index) => (
                      <li key={index} className="text-sm">
                        <strong>{issue.name}:</strong> {issue.message}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Contingency Plans */}
              <div>
                <h3 className="font-semibold mb-3">🛡️ Contingency Plans</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {report.contingencyPlans.map((plan, index) => (
                    <div
                      key={index}
                      className={`p-3 rounded-lg border ${
                        isDark ? 'border-gray-600 bg-gray-800' : 'border-gray-200 bg-gray-50'
                      }`}
                    >
                      <div className="flex items-start space-x-2">
                        <span className="text-orange-500 mt-1">🛡️</span>
                        <span className="text-sm">{plan}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Emergency Contacts */}
              <div className={`p-4 rounded-lg border ${
                isDark ? 'border-gray-600 bg-gray-800' : 'border-gray-200 bg-gray-50'
              }`}>
                <h3 className="font-semibold mb-2">📞 Emergency Checklist</h3>
                <ul className="space-y-1 text-sm">
                  <li>✓ Backup demo video ready and tested</li>
                  <li>✓ Offline mode enabled and functional</li>
                  <li>✓ Alternative device available</li>
                  <li>✓ Demo script memorized</li>
                  <li>✓ Technical support contact available</li>
                  <li>✓ Presentation slides as backup</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Loading State */}
      {isChecking && (
        <Card>
          <CardContent className="text-center py-8">
            <div className="animate-spin text-4xl mb-4">🔄</div>
            <p>Running comprehensive environment check...</p>
          </CardContent>
        </Card>
      )}

      {/* No Report State */}
      {!report && !isChecking && (
        <Card>
          <CardContent className="text-center py-8">
            <div className="text-4xl mb-4">🎯</div>
            <p className="mb-4">Ready to check your demo environment?</p>
            <Button variant="primary" onClick={runCheck}>
              🔍 Run Environment Check
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default DemoEnvironmentPanel