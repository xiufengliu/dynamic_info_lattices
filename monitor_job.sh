#!/bin/bash
# Monitor Dynamic Information Lattices training job

JOB_ID=25710198

echo "Monitoring DIL Training Job: $JOB_ID"
echo "=================================="

# Check job status
echo "Job Status:"
bjobs $JOB_ID

echo ""
echo "Detailed Job Info:"
bjobs -l $JOB_ID | head -20

echo ""
echo "Output Log (last 20 lines):"
if [ -f "logs/dil_train_${JOB_ID}.out" ]; then
    tail -20 logs/dil_train_${JOB_ID}.out
else
    echo "Output log not yet created (job not started)"
fi

echo ""
echo "Error Log (last 10 lines):"
if [ -f "logs/dil_train_${JOB_ID}.err" ]; then
    tail -10 logs/dil_train_${JOB_ID}.err
else
    echo "Error log not yet created (job not started)"
fi

echo ""
echo "To monitor in real-time:"
echo "  bjobs $JOB_ID"
echo "  tail -f logs/dil_train_${JOB_ID}.out"
echo "  bpeek $JOB_ID"
