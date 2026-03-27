"""Quick progress checker for the coarse sweep."""
import sqlite3
import json
import sys

DB = "/home/cisco/structural_intervention_harness/experiment.db"

def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    
    total = conn.execute('SELECT COUNT(*) FROM trials').fetchone()[0]
    non_sanity = conn.execute("SELECT COUNT(*) FROM trials WHERE category != 'SANITY'").fetchone()[0]
    
    print(f"=== Coarse Sweep Progress: {non_sanity}/612 trials ===\n")
    
    # By category
    rows = conn.execute("""
        SELECT category, COUNT(*) as n, 
               AVG(CASE WHEN is_degenerate=0 THEN accuracy_delta END) as avg_delta,
               MIN(CASE WHEN is_degenerate=0 THEN accuracy_delta END) as min_delta, 
               MAX(CASE WHEN is_degenerate=0 THEN accuracy_delta END) as max_delta,
               SUM(is_degenerate) as degen,
               MIN(CASE WHEN is_degenerate=0 THEN mcnemar_p_value END) as best_p
        FROM trials WHERE category != 'SANITY'
        GROUP BY category ORDER BY category
    """).fetchall()
    
    print(f"{'Cat':<6} {'N':>4} {'Avg Δ':>7} {'Min Δ':>7} {'Max Δ':>7} {'Deg':>4} {'Best p':>8}")
    print("-" * 50)
    for r in rows:
        avg = f"{r['avg_delta']:+.1f}" if r['avg_delta'] else "N/A"
        mn = f"{r['min_delta']:+.1f}" if r['min_delta'] else "N/A"
        mx = f"{r['max_delta']:+.1f}" if r['max_delta'] else "N/A"
        bp = f"{r['best_p']:.4f}" if r['best_p'] else "N/A"
        print(f"{r['category']:<6} {r['n']:>4} {avg:>7} {mn:>7} {mx:>7} {r['degen']:>4.0f} {bp:>8}")
    
    # Significant results
    sig = conn.execute("""
        SELECT trial_id, category, accuracy_delta, mcnemar_p_value, 
               items_flipped_to_correct, items_flipped_to_incorrect
        FROM trials WHERE is_degenerate=0 AND mcnemar_p_value < 0.05
        ORDER BY mcnemar_p_value
    """).fetchall()
    
    print(f"\n=== Significant (p<0.05 pre-BH): {len(sig)} ===")
    for r in sig:
        print(f"  {r['trial_id']:<35} {r['category']:<5} Δ={r['accuracy_delta']:+.1f}pp  p={r['mcnemar_p_value']:.4f}  +{r['items_flipped_to_correct']}/-{r['items_flipped_to_incorrect']}")
    
    # Top 5 improvements
    top = conn.execute("""
        SELECT trial_id, category, accuracy_delta, mcnemar_p_value
        FROM trials WHERE is_degenerate=0 
        ORDER BY accuracy_delta DESC LIMIT 5
    """).fetchall()
    print(f"\n=== Top 5 improvements ===")
    for r in top:
        print(f"  {r['trial_id']:<35} {r['category']:<5} Δ={r['accuracy_delta']:+.1f}pp  p={r['mcnemar_p_value']:.4f}")
    
    # Top 5 degradations
    bot = conn.execute("""
        SELECT trial_id, category, accuracy_delta, mcnemar_p_value
        FROM trials WHERE is_degenerate=0 
        ORDER BY accuracy_delta ASC LIMIT 5
    """).fetchall()
    print(f"\n=== Top 5 degradations ===")
    for r in bot:
        print(f"  {r['trial_id']:<35} {r['category']:<5} Δ={r['accuracy_delta']:+.1f}pp  p={r['mcnemar_p_value']:.4f}")
    
    conn.close()

if __name__ == "__main__":
    main()
