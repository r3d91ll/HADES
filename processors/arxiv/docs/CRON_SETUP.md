# Cron Job Setup for Daily ArXiv Updates

## Current Status

**DISABLED** - The cron job is currently disabled while the database rebuild with Jina v4 is in progress.

## Cron Configuration

### Original Entry (Now Disabled)
```bash
#DISABLED_UNTIL_V4_READY: 0 1 * * * ARANGO_PASSWORD='[REDACTED]' cd /home/todd/olympus/HADES/processors/arxiv && poetry run python3 scripts/daily_arxiv_update.py >> /home/todd/olympus/HADES/processors/logs/cron_arxiv.log 2>&1
```

### Schedule Details
- **Time**: 01:00 UTC daily
- **Script**: `scripts/daily_arxiv_update.py`
- **Log File**: `/home/todd/olympus/HADES/processors/logs/cron_arxiv.log`
- **Environment**: Runs with Poetry virtual environment

## Re-enabling the Cron Job

After the database rebuild completes:

### 1. Test the Updated Script
```bash
# Test with yesterday's papers
export ARANGO_PASSWORD='your_password'
cd /home/todd/olympus/HADES/processors/arxiv
poetry run python3 scripts/daily_arxiv_update.py --days-back 1

# Check for specific date range if needed
poetry run python3 scripts/daily_arxiv_update.py --days-back 7
```

### 2. Re-enable Cron Job
```bash
# Edit crontab
crontab -e

# Uncomment the line (remove #DISABLED_UNTIL_V4_READY:)
0 1 * * * ARANGO_PASSWORD='your_password' cd /home/todd/olympus/HADES/processors/arxiv && poetry run python3 scripts/daily_arxiv_update.py >> /home/todd/olympus/HADES/processors/logs/cron_arxiv.log 2>&1
```

### 3. Monitor Logs
```bash
# Watch daily update logs
tail -f /home/todd/olympus/HADES/processors/logs/cron_arxiv.log

# Check for errors
grep ERROR /home/todd/olympus/HADES/processors/logs/cron_arxiv.log
```

## What the Updated Script Does

The `daily_arxiv_update.py` has been updated for Jina v4:

1. **Fetches New Papers**: Uses ArXiv API to get yesterday's papers
2. **Generates Embeddings**: Creates 2048-dim Jina v4 embeddings
3. **Updates Database**: Inserts new papers or updates existing ones
4. **Preserves History**: Maintains update logs in `daily_updates` collection

### Key Features
- Automatic Jina v4 embedding generation
- Graceful fallback if GPU unavailable (stores without embeddings)
- Batch processing for efficiency
- Update logging for audit trail
- Rate limiting for ArXiv API compliance

## Performance Expectations

- **Papers per day**: ~500-1500 (varies)
- **Processing time**: 5-15 minutes
- **GPU usage**: Uses GPU 1 by default (GPU 0 reserved for other tasks)
- **Embedding rate**: ~30-50 papers/second with GPU

## Troubleshooting

### If Cron Doesn't Run
```bash
# Check cron service
systemctl status cron

# Check crontab
crontab -l

# Test manually
cd /home/todd/olympus/HADES/processors/arxiv
ARANGO_PASSWORD='your_password' poetry run python3 scripts/daily_arxiv_update.py
```

### If Embeddings Fail
```bash
# Run without embeddings
poetry run python3 scripts/daily_arxiv_update.py --skip-embeddings

# Later, batch embed missing papers
poetry run python3 scripts/rebuild_dual_gpu.py --only-missing-embeddings
```

### Database Connection Issues
```bash
# Test database connection
python3 -c "
from arango import ArangoClient
import os
client = ArangoClient(hosts='http://192.168.1.69:8529')
db = client.db('academy_store', username='root', password=os.environ['ARANGO_PASSWORD'])
print(f'Collections: {db.collections()}')
"
```

## Catchup After Downtime

If the cron job is disabled for multiple days:

```bash
# Catch up last 7 days
for i in {1..7}; do
    echo "Processing $i days ago..."
    poetry run python3 scripts/daily_arxiv_update.py --days-back $i
    sleep 5
done
```

## Future Improvements

1. **Parallel Processing**: Use both GPUs for faster embedding
2. **Incremental PDF Processing**: Download and process full PDFs for high-impact papers
3. **Citation Tracking**: Build citation networks from references
4. **Category Filtering**: Focus on specific research areas
5. **Duplicate Detection**: Better handling of versioned papers

---

*Note: Remember to update the ARANGO_PASSWORD in the cron entry when re-enabling!*