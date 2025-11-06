/**
 * E2E Test Execution Script
 * =========================
 * 
 * Comprehensive end-to-end test execution
 */

import E2ETestRunner from './TestRunner'

async function runE2ETests() {
  console.log('🚀 Starting SignEase E2E Integration Tests...\n')
  
  const runner = new E2ETestRunner()
  
  try {
    const report = await runner.runAllTests()
    
    // Determine exit code based on results
    const criticalFailures = report.suites.some(suite => suite.passRate < 0.7)
    const accuracyMet = report.requirements.actualAccuracy >= report.requirements.accuracyTarget
    const performanceMet = report.requirements.actualPerformance <= report.requirements.performanceTarget
    
    if (criticalFailures || !accuracyMet || !performanceMet) {
      console.log('❌ E2E Tests FAILED - Critical issues detected')
      process.exit(1)
    } else if (report.summary.overallPassRate < 0.9) {
      console.log('⚠️ E2E Tests PASSED with warnings - Some issues detected')
      process.exit(0)
    } else {
      console.log('✅ E2E Tests PASSED - All requirements met!')
      process.exit(0)
    }
    
  } catch (error) {
    console.error('💥 E2E Test execution failed:', error)
    process.exit(1)
  }
}

// Run if called directly
if (require.main === module) {
  runE2ETests()
}

export default runE2ETests