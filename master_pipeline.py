# Master pipeline: URL generation → LLM extraction → GEE enrichment → failure retry.
# Stages 2 and 3 are decoupled so GEE slowdowns don't stall extraction.
# Run Builder_Matching + Builder_Combined afterwards to refresh the combined datasets.

from datetime import date

from Relevant_News_Retrieval.pipeline import start_pipeline as run_stage1
from Builder_GDELT.pipelineRunner import run_pipeline_date_range as run_stage2
from Builder_GDELT.run_enrichment import run_enrichment as run_stage3
from Builder_GDELT.helper_scripts.pipeline.reenrich_failed import scan_all_days, reenrich_day, ENRICHED_ROOT


# ---- Set your dates here ---- #
#YYYY M DD
START_DATE = date(2017, 8, 1)
END_DATE   = date(2017, 12, 31)

# YYYY  M  DD
START_DATE = date(2017, 1, 21)
END_DATE   = date(2017, 12, 31)

POSTPROCESS = True


def run_full_pipeline():
    #print("Running Stage 1...")
    run_stage1(START_DATE.strftime("%Y%m%d"),END_DATE.strftime("%Y%m%d"))

    print("Running Stage 2...")
    #run_stage2(START_DATE, END_DATE, postprocess=POSTPROCESS)

    print("\nRunning Stage 3: GEE enrichment...")
    run_stage3(date_from=START_DATE, date_to=END_DATE)

    print("\nRunning Stage 4: Re-enriching any GEE failures...")
    # Limit the scan to the current run's date range — scanning the full archive
    # on every run gets increasingly slow as the dataset grows.
    #run_days = []
    #d = START_DATE
    #while d <= END_DATE:
        #run_days.append(d.strftime("%Y%m%d"))
        #d = date.fromordinal(d.toordinal() + 1)
    #summaries = scan_all_days(ENRICHED_ROOT, days=run_days)
    #if summaries:
       # total = sum(len(s["gee_failed"]) + len(s["no_date"]) for s in summaries)
        #print(f"  Found {total} fixable failures across {len(summaries)} days")
        #for s in summaries:
        #    reenrich_day(s)
   #else:
     #   print("  No fixable failures found.")

    #print("Done.")


if __name__ == "__main__":
    run_full_pipeline()
