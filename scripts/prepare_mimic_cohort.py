import os
import random
import datetime
import pandas as pd
import numpy as np

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Clinical conditions and key risk factors for hospital readmission
HIGH_RISK_TEMPLATES = [
    {
        "chief_complaint": "Acute exacerbation of congestive heart failure and shortness of breath.",
        "hpi": "Patient is a 72-year-old male with a history of severe systolic heart failure (EF 25%), hypertension, and stage 3 chronic kidney disease presenting with progressive dyspnea, 3+ bilateral lower extremity edema, and orthopnea over 4 days. Patient admits to non-compliance with low-sodium diet and missed several doses of Lasix.",
        "course": "Patient was treated with intravenous furosemide with partial diuresis. Elevated BNP at 1450 pg/mL. Serum creatinine rose from 1.8 to 2.6 mg/dL during diuresis. Oxygen saturation dropped to 89% on room air requiring 2L nasal cannula. Patient remains fluid overloaded upon discharge with persistent bilateral lung crackles.",
        "diagnoses": "Acute decompensated heart failure, Stage 3 Chronic Kidney Disease, Hypertension, Non-compliance with medical regimen.",
        "instructions": "Follow up with cardiology in 3 days. Weigh daily and call if weight increases by >2 lbs in 24 hours. Strict sodium restriction <2g/day.",
        "readmission_risk": 1
    },
    {
        "chief_complaint": "COPD exacerbation, severe dyspnea, and productive cough.",
        "hpi": "68-year-old female with severe COPD (FEV1 40%), active 40 pack-year smoking history, and frequent hospitalizations (3 in past 6 months) presenting with worsening shortness of breath, wheezing, and purulent sputum. Patient ran out of home oxygen 2 days prior.",
        "course": "Started on IV methylprednisolone, nebulized albuterol/ipratropium, and azithromycin. Patient continues to have frequent paroxysmal coughing fits and resting hypoxemia (SpO2 88% on room air). Discharged with a 5-day oral prednisone burst and home oxygen.",
        "diagnoses": "Severe acute COPD exacerbation, Chronic respiratory failure, Tobacco use disorder, Frequent readmitter status.",
        "instructions": "Follow up with pulmonology within 5 days. Continue oxygen at 2L/min continuous. Call clinic if sputum color changes or dyspnea worsens.",
        "readmission_risk": 1
    },
    {
        "chief_complaint": "Diabetic ketoacidosis and uncontrolled hyperglycemia.",
        "hpi": "54-year-old male with long-standing Type 2 Diabetes Mellitus (HbA1c 12.4%), peripheral neuropathy, and recurrent hypoglycemic/hyperglycemic admissions. Presents with nausea, vomiting, abdominal pain, and blood glucose of 580 mg/dL.",
        "course": "Admitted to ICU for insulin drip protocol and aggressive IV hydration. Anion gap closed after 24 hours. Patient reports inability to afford Lantus prescription at home. Social work consulted. Discharge blood glucose remains labile (220-310 mg/dL).",
        "diagnoses": "Diabetic Ketoacidosis, Uncontrolled Type 2 Diabetes Mellitus, Medication non-adherence due to financial barrier.",
        "instructions": "Follow up with endocrinology in 1 week. Check blood glucose 4 times daily. Social work assistance provided for medication access.",
        "readmission_risk": 1
    },
    {
        "chief_complaint": "Severe sepsis secondary to urinary tract infection and acute kidney injury.",
        "hpi": "81-year-old female resident of skilled nursing facility with history of dementia, recurrent UTIs, and neurogenic bladder presenting with lethargy, fever of 102.4F, tachycardia, and hypotension (BP 84/50).",
        "course": "Received 3L normal saline boluses and started on broad-spectrum IV ceftriaxone. Blood cultures positive for E. coli. Acute kidney injury resolved, but patient remains frail, bedbound, with poor oral intake and ongoing urinary incontinence via indwelling Foley catheter.",
        "diagnoses": "Sepsis secondary to UTI, E. coli bacteremia, Frailty syndrome, Indwelling urinary catheter.",
        "instructions": "Complete 10-day oral cefdinir course. Nursing facility to monitor urine output and temperature twice daily.",
        "readmission_risk": 1
    },
    {
        "chief_complaint": "Recurrent gastrointestinal bleeding and severe anemia.",
        "hpi": "75-year-old male on chronic apixaban for atrial fibrillation and history of peptic ulcer disease presenting with dark melenic stools, lightheadedness, and hemoglobin of 6.2 g/dL.",
        "course": "Apixaban held. Transfused 3 units PRBCs with post-transfusion Hgb 8.8 g/dL. Esophagogastroduodenoscopy (EGD) revealed diffuse antral gastritis with oozing ulcer. Cauterized and clipped. High risk for recurrent bleed if anticoagulation resumed.",
        "diagnoses": "Upper GI Bleeding, Severe acute blood loss anemia, Gastric ulcer, Atrial fibrillation on anticoagulation.",
        "instructions": "Gastroenterology follow-up in 7 days. Hold Eliquis for 5 days then re-evaluate with cardiologist. Report dark or tarry stools immediately.",
        "readmission_risk": 1
    }
]

LOW_RISK_TEMPLATES = [
    {
        "chief_complaint": "Elective laparoscopic cholecystectomy for symptomatic cholelithiasis.",
        "hpi": "42-year-old female with history of biliary colic presented for elective outpatient laparoscopic cholecystectomy. Preoperative labs and ECG were unremarkable.",
        "course": "Procedure completed without complication. Minimal intraoperative blood loss. Gallbladder extracted intact. Patient tolerated clear liquid diet postoperatively and pain was well-controlled on oral acetaminophen/ibuprofen.",
        "diagnoses": "Symptomatic cholelithiasis, Status post uncomplicated laparoscopic cholecystectomy.",
        "instructions": "Follow up with general surgery in 2 weeks. Resume normal diet as tolerated. Avoid heavy lifting >10 lbs for 2 weeks.",
        "readmission_risk": 0
    },
    {
        "chief_complaint": "Uncomplicated right inguinal hernia repair.",
        "hpi": "50-year-old healthy male presenting for elective mesh repair of a reducible right inguinal hernia causing mild discomfort with heavy lifting.",
        "course": "Open mesh repair performed under general anesthesia without intraoperative complications. Patient voided spontaneously and ambulated well in PACU. Pain controlled with oral analgesics.",
        "diagnoses": "Right inguinal hernia, Status post open mesh hernia repair.",
        "instructions": "Follow up with surgical clinic in 10-14 days. Keep incision clean and dry. No lifting over 15 pounds for 4 weeks.",
        "readmission_risk": 0
    },
    {
        "chief_complaint": "Community-acquired pneumonia, clinically resolved.",
        "hpi": "35-year-old male with no significant past medical history presented 4 days ago with fever, right lower lobe consolidation on chest X-ray, and productive cough.",
        "course": "Treated with a 5-day course of oral azithromycin. Fever resolved within 24 hours of presentation. Oxygen saturation 98% on room air throughout admission. Cough markedly improved.",
        "diagnoses": "Community-acquired pneumonia, resolved.",
        "instructions": "Complete remaining 2 days of oral antibiotics. Rest at home for 3 days before returning to work. Follow up with primary care doctor in 2 weeks.",
        "readmission_risk": 0
    },
    {
        "chief_complaint": "Right distal radius fracture following low-energy fall.",
        "hpi": "28-year-old female fell while rollerblading, landing on outstretched right hand. Closed, minimally displaced distal radius fracture confirmed on imaging.",
        "course": "Successful closed reduction performed under hematoma block in Emergency Department. Short arm plaster cast applied with good alignment on post-reduction X-rays. Neurovascular exam intact distal to cast.",
        "diagnoses": "Closed right distal radius fracture, post-reduction.",
        "instructions": "Follow up with orthopedics in 7-10 days for cast check and repeat X-rays. Keep cast dry and elevate arm above heart level.",
        "readmission_risk": 0
    },
    {
        "chief_complaint": "Mild acute gastroenteritis with mild dehydration.",
        "hpi": "29-year-old male with 2-day history of watery diarrhea and emesis following consumption of street food. No blood in stool, no fever.",
        "course": "Received 1L normal saline hydration. Electrolytes within normal limits. Stool studies negative for bacterial pathogens. Symptoms completely settled with oral rehydration solution and light diet.",
        "diagnoses": "Acute viral gastroenteritis, Resolved dehydration.",
        "instructions": "Follow bland diet (BRAT diet) for 24-48 hours. Maintain hydration. Follow up with PCP as needed.",
        "readmission_risk": 0
    }
]

def generate_mimic_discharge_note(template, subject_id, hadm_id, admittime, dischtime):
    note_text = f"""
================================================================================
MIMIC-III / MIMIC-IV DISCHARGE SUMMARY
================================================================================
PATIENT ID: {subject_id}
ADMISSION ID: {hadm_id}
ADMISSION DATE: {admittime.strftime('%Y-%m-%d %H:%M')}
DISCHARGE DATE: {dischtime.strftime('%Y-%m-%d %H:%M')}
SERVICE: MEDICINE / SURGERY

CHIEF COMPLAINT:
{template['chief_complaint']}

HISTORY OF PRESENT ILLNESS:
{template['hpi']}

HOSPITAL COURSE & CLINICAL METRICS:
{template['course']}

DISCHARGE DIAGNOSES:
{template['diagnoses']}

DISCHARGE INSTRUCTIONS & FOLLOW-UP:
{template['instructions']}
================================================================================
""".strip()
    return note_text


def build_dataset(num_samples=250):
    records = []
    base_date = datetime.datetime(2025, 1, 1, 8, 0)
    
    for i in range(num_samples):
        subject_id = 10000 + i
        hadm_id = 200000 + i
        
        # 50% high risk, 50% low risk
        is_high_risk = (i % 2 == 1)
        if is_high_risk:
            template = random.choice(HIGH_RISK_TEMPLATES)
            readmitted = 1
        else:
            template = random.choice(LOW_RISK_TEMPLATES)
            readmitted = 0
            
        days_offset = random.randint(1, 180)
        admittime = base_date + datetime.timedelta(days=days_offset, hours=random.randint(0, 12))
        length_of_stay = random.randint(1, 7) if not is_high_risk else random.randint(3, 12)
        dischtime = admittime + datetime.timedelta(days=length_of_stay, hours=random.randint(1, 10))
        
        note = generate_mimic_discharge_note(template, subject_id, hadm_id, admittime, dischtime)
        
        records.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "admittime": admittime.strftime('%Y-%m-%d %H:%M:%S'),
            "dischtime": dischtime.strftime('%Y-%m-%d %H:%M:%S'),
            "chief_complaint": template["chief_complaint"],
            "discharge_note": note,
            "readmitted_within_30d": readmitted,
        })
        
    df = pd.DataFrame(records)
    
    # Stratified Train (80%) / Test (20%) split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["readmitted_within_30d"]
    )
    
    df["split"] = "train"
    df.loc[test_df.index, "split"] = "test"
    
    # Save as parquet and csv
    parquet_path = os.path.join(DATA_DIR, "mimic_readmission_cohort.parquet")
    csv_path = os.path.join(DATA_DIR, "mimic_readmission_cohort.csv")
    
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset created successfully!")
    print(f"Total Cohort Size: {len(df)}")
    print(f"Train Set Size: {len(df[df['split'] == 'train'])} (Readmitted: {df[df['split'] == 'train']['readmitted_within_30d'].sum()})")
    print(f"Test Set Size:  {len(df[df['split'] == 'test'])} (Readmitted: {df[df['split'] == 'test']['readmitted_within_30d'].sum()})")
    print(f"Saved to:\n  - {parquet_path}\n  - {csv_path}")

if __name__ == "__main__":
    build_dataset()
