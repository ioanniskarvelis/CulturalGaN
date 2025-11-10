# 🤖 Using Anthropic Claude for Symbolic Analysis

## Quick Setup

### 1. Get Your API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to "API Keys"
4. Create a new API key
5. Copy the key (starts with `sk-ant-...`)

### 2. Set the API Key

**Option A: Environment Variable (Recommended)**

Windows PowerShell:
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-api03-..."
```

Windows CMD:
```cmd
set ANTHROPIC_API_KEY=sk-ant-api03-...
```

Linux/Mac:
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

**Option B: Create .env file**

Create a file named `.env` in your project root:
```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

### 3. Run Phase 2 with Claude

**Test with 10 images:**
```bash
python scripts/run_phase2.py --use-vision --limit 10
```

**Process all images:**
```bash
python scripts/run_phase2.py --use-vision
```

---

## Why Claude?

### Advantages of Claude 3.5 Sonnet:
- ✅ **Better cultural understanding** - More nuanced analysis of symbolism
- ✅ **Longer context window** - 200K tokens (handles detailed descriptions)
- ✅ **Cost-effective** - ~$0.003 per image vs $0.01 for GPT-4V
- ✅ **High quality vision** - Excellent at analyzing patterns and details
- ✅ **Structured outputs** - Better at following formatting instructions

### Cost Comparison:
| Model | Cost per Image | 475 Images Total |
|-------|---------------|------------------|
| **Claude 3.5 Sonnet** | ~$0.003-0.005 | **~$1.50-2.50** |
| GPT-4 Vision | ~$0.01-0.02 | ~$5-10 |
| Fallback (free) | $0 | $0 |

---

## Model Details

**Default Model**: `claude-3-5-sonnet-20241022`

**Alternative Models**:
- `claude-3-5-sonnet-20241022` - Latest, best quality (recommended)
- `claude-3-opus-20240229` - Highest quality, slower, more expensive
- `claude-3-sonnet-20240229` - Older version
- `claude-3-haiku-20240307` - Fastest, cheapest, lower quality

### Change Model:
```bash
python scripts/run_phase2.py --use-vision --model claude-3-opus-20240229
```

---

## Verify Installation

Test if Anthropic is set up correctly:

```python
# test_anthropic.py
import anthropic
import os

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("❌ API key not found!")
    print("Set ANTHROPIC_API_KEY environment variable")
else:
    print(f"✅ API key found: {api_key[:20]}...")
    
    # Test connection
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(f"✅ Connection successful!")
    print(f"Response: {message.content[0].text}")
```

Run:
```bash
python test_anthropic.py
```

---

## Example: Phase 2 with Claude

### Full workflow:

```bash
# 1. Set API key
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# 2. Test with 10 images first
python scripts/run_phase2.py --use-vision --limit 10

# 3. Check the results
cat data/annotations/symbolic_analysis_report.json

# 4. If satisfied, process all images
python scripts/run_phase2.py --use-vision

# 5. Monitor progress (saved every 10 images)
# You can stop and restart anytime - it will skip existing annotations
```

---

## Troubleshooting

### Error: "anthropic package not found"
```bash
pip install anthropic
```

### Error: "API key not found"
Check that the environment variable is set:
```bash
# Windows PowerShell
echo $env:ANTHROPIC_API_KEY

# Linux/Mac
echo $ANTHROPIC_API_KEY
```

### Error: "Rate limit exceeded"
Claude has rate limits. The script automatically handles this by:
- Processing one image at a time
- Saving progress every 10 images
- You can resume anytime

If you hit limits, just wait a few minutes and resume:
```bash
python scripts/run_phase2.py --use-vision
# It will skip already-processed images automatically
```

### Error: "Invalid API key"
Make sure:
1. You copied the full key (starts with `sk-ant-`)
2. The key is active in your Anthropic console
3. You have credits/billing set up

---

## Output Quality with Claude

Claude 3.5 Sonnet provides excellent analysis. Example output:

```json
{
  "pattern_type": "geometric",
  "geometric_structure": "Bilateral symmetry with repeating Greek key (meander) pattern. Angular 90-degree turns create continuous interlocking design. Four-fold rotational symmetry around central axis.",
  "cultural_symbolism": "The Greek key represents the eternal flow of life and unity. In Cycladic tradition, this pattern symbolizes the labyrinth and the journey of life. The continuous line without beginning or end represents immortality and the infinite nature of existence.",
  "historical_context": "Common in Cycladic art from 3000-2000 BCE, later adopted across Greek civilization. Frequently found on pottery borders, textile edges, and architectural friezes. This specific variant suggests Late Bronze Age origin.",
  "color_significance": "Black on white represents the duality of existence - light and darkness, life and death. Traditional Cycladic color scheme using natural pigments: charcoal black and limestone white.",
  "preservation_notes": "Critical to preserve: 1) Exact 90-degree angles, 2) Uniform line width, 3) Perfect spacing between parallel lines, 4) Continuous unbroken path, 5) Bilateral symmetry axis",
  "authenticity_score": 0.95,
  "key_features": ["meander", "bilateral_symmetry", "geometric_precision", "traditional_cycladic_colors", "continuous_line"]
}
```

---

## Next Steps After Phase 2

Once symbolic analysis is complete:

```bash
# Train the GAN
python scripts/train_gan.py
```

The embeddings created from Claude's analysis will condition the GAN to generate authentic Greek motifs!

---

## Getting Credits

**Free Tier**: Anthropic typically provides $5 free credits for new accounts.

**For this project**: 
- 475 images × $0.004 ≈ **$1.90**
- Fits within free tier!

**Paid Plans**: If you need more, Claude Pro is $20/month with higher limits.

---

**Ready to go?** Set your API key and run:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key
python scripts/run_phase2.py --use-vision --limit 10
```

Good luck! 🚀

