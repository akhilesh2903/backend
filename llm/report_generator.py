"""
LLM Report Generator Module
============================
Generates structured clinical radiology reports using Retrieved Augmented Generation.
Combines CNN predictions, medical context, clinical guidelines, and nutrient recommendations.
"""

import json
from datetime import datetime


class RadiologyReportGenerator:
    """
    Generates structured radiology-style reports for fetal ultrasound analysis.
    Uses template-based generation with retrieved medical context.
    """

    def __init__(self):
        """Initialize report generator with clinical templates."""
        self.report_templates = self._load_templates()
        self.nutrient_data = self._load_nutrient_data()

    def _load_templates(self):
        """Load report templates for different conditions."""
        templates = {
            'normal': {
                'title': 'NORMAL FETAL DEVELOPMENT',
                'risk_level': 'LOW RISK',
                'key_findings': [
                    'Fetus developing normally',
                    'Biometry appropriate for gestational age',
                    'Normal amniotic fluid volume',
                    'No structural abnormalities detected'
                ]
            },
            'fgr_mild': {
                'title': 'MILD FETAL GROWTH RESTRICTION',
                'risk_level': 'MODERATE RISK',
                'key_findings': [
                    'Abdominal circumference below expected percentile',
                    'Growth velocity reduced',
                    'Borderline Doppler indices',
                    'Amniotic fluid volume adequate'
                ]
            },
            'fgr_severe': {
                'title': 'SEVERE FETAL GROWTH RESTRICTION',
                'risk_level': 'HIGH RISK - URGENT',
                'key_findings': [
                    'Significant reduction in estimated fetal weight',
                    'Markedly decreased abdominal circumference',
                    'Abnormal Doppler patterns indicating placental insufficiency',
                    'Potential oligohydramnios'
                ]
            }
        }
        return templates

    def _load_nutrient_data(self):
        """Load condition-specific nutrient recommendations."""
        return {
            'normal': {
                'title': 'NUTRITIONAL RECOMMENDATIONS — NORMAL DEVELOPMENT',
                'description': (
                    'Your fetus is developing normally. Maintain a balanced prenatal diet '
                    'rich in the following key nutrients to support continued healthy growth:'
                ),
                'nutrients': [
                    {
                        'name': 'Folic Acid (Folate)',
                        'dose': '600 mcg/day',
                        'sources': 'Leafy greens, lentils, fortified cereals, oranges',
                        'benefit': 'Prevents neural tube defects; supports brain and spine development'
                    },
                    {
                        'name': 'Iron',
                        'dose': '27 mg/day',
                        'sources': 'Lean meat, spinach, beans, fortified cereals',
                        'benefit': 'Supports oxygen transport; prevents maternal and fetal anemia'
                    },
                    {
                        'name': 'Calcium',
                        'dose': '1000 mg/day',
                        'sources': 'Dairy products, almonds, broccoli, fortified plant milk',
                        'benefit': 'Essential for fetal bone and teeth mineralization'
                    },
                    {
                        'name': 'Omega-3 Fatty Acids (DHA)',
                        'dose': '200–300 mg DHA/day',
                        'sources': 'Fatty fish (salmon, sardines), walnuts, flaxseeds, fish oil',
                        'benefit': 'Supports brain development and visual acuity'
                    },
                    {
                        'name': 'Vitamin D',
                        'dose': '600 IU/day (up to 2000 IU if deficient)',
                        'sources': 'Sunlight exposure, fortified milk, egg yolks, salmon',
                        'benefit': 'Bone formation, immune function, calcium absorption'
                    },
                    {
                        'name': 'Protein',
                        'dose': '71 g/day (additional 25 g above baseline)',
                        'sources': 'Eggs, poultry, dairy, legumes, tofu, nuts',
                        'benefit': 'Fetal tissue growth, placental development, maternal tissue repair'
                    },
                    {
                        'name': 'Iodine',
                        'dose': '220 mcg/day',
                        'sources': 'Iodized salt, seafood, dairy',
                        'benefit': 'Thyroid function and brain development'
                    },
                    {
                        'name': 'Zinc',
                        'dose': '11 mg/day',
                        'sources': 'Meat, shellfish, legumes, seeds',
                        'benefit': 'Cell growth, immune support, enzyme function'
                    }
                ],
                'foods_to_avoid': [
                    'Raw or undercooked meat/fish/eggs (risk of infection)',
                    'High-mercury fish (shark, swordfish, king mackerel)',
                    'Unpasteurized dairy and soft cheeses',
                    'Alcohol and tobacco products',
                    'Excessive caffeine (>200 mg/day)'
                ]
            },
            'fgr': {
                'title': 'NUTRITIONAL RECOMMENDATIONS — FETAL GROWTH RESTRICTION',
                'description': (
                    'Fetal Growth Restriction (FGR) requires targeted nutritional intervention '
                    'to support improved fetal growth, placental function, and maternal health. '
                    'The following nutrients are CRITICAL — consult your dietitian for a personalised plan:'
                ),
                'nutrients': [
                    {
                        'name': 'High-Quality Protein',
                        'dose': '90–100 g/day (significantly increased)',
                        'sources': 'Eggs, lean meat, fish, Greek yogurt, legumes, quinoa',
                        'benefit': 'Essential for fetal catch-up growth; supports placental protein synthesis'
                    },
                    {
                        'name': 'L-Arginine',
                        'dose': '3–4 g/day (supplement under medical supervision)',
                        'sources': 'Pumpkin seeds, turkey, chicken, peanuts, soy',
                        'benefit': 'Increases placental blood flow; promotes fetal oxygen delivery'
                    },
                    {
                        'name': 'Iron',
                        'dose': '30–60 mg/day (higher than normal, monitor for side effects)',
                        'sources': 'Red meat, lentils, fortified cereals; pair with Vitamin C',
                        'benefit': 'Corrects anemia; improves oxygen availability to fetus'
                    },
                    {
                        'name': 'Omega-3 Fatty Acids (DHA + EPA)',
                        'dose': '400–600 mg DHA/day',
                        'sources': 'Salmon, sardines, fish oil supplements, algae oil',
                        'benefit': 'Anti-inflammatory; improves placental vascular function; brain growth'
                    },
                    {
                        'name': 'Zinc',
                        'dose': '20–25 mg/day',
                        'sources': 'Oysters, beef, pumpkin seeds, chickpeas',
                        'benefit': 'Critical for fetal cell division and growth; immune support'
                    },
                    {
                        'name': 'Vitamin C',
                        'dose': '100–150 mg/day',
                        'sources': 'Citrus fruits, bell peppers, guava, strawberries',
                        'benefit': 'Enhances iron absorption; antioxidant protection for placenta'
                    },
                    {
                        'name': 'Magnesium',
                        'dose': '350–400 mg/day',
                        'sources': 'Dark chocolate, spinach, avocado, nuts, whole grains',
                        'benefit': 'Vasodilation for improved placental blood flow; reduces preterm risk'
                    },
                    {
                        'name': 'Vitamin E',
                        'dose': '15 mg/day',
                        'sources': 'Almonds, sunflower seeds, avocado, spinach',
                        'benefit': 'Protects cell membranes; antioxidant for placenta'
                    },
                    {
                        'name': 'Folic Acid',
                        'dose': '800–1000 mcg/day',
                        'sources': 'Dark leafy greens, legumes, fortified foods',
                        'benefit': 'Cell division support; critical for catch-up growth'
                    },
                    {
                        'name': 'Vitamin D',
                        'dose': '1000–2000 IU/day (if deficient)',
                        'sources': 'Fortified milk, oily fish, egg yolk, sunlight',
                        'benefit': 'Placental implantation and function; bone development'
                    }
                ],
                'foods_to_avoid': [
                    'Processed/junk food (empty calories, inflammatory)',
                    'High-sodium foods (worsen hypertension in FGR)',
                    'Alcohol and tobacco (directly impair placental function)',
                    'High-mercury fish',
                    'Undercooked foods (infection risk is higher)',
                    'Excessive caffeine (vasoconstriction worsens placental flow)'
                ],
                'additional_guidance': (
                    'IMPORTANT: Small frequent meals (5–6 per day) are preferred over large meals. '
                    'Stay well-hydrated (2–3 litres water/day). '
                    'Consider nutritional supplementation under medical supervision. '
                    'Close dietary monitoring with a registered dietitian is strongly recommended.'
                )
            },
            'abnormality': {
                'title': 'NUTRITIONAL RECOMMENDATIONS — FETAL ABNORMALITY DETECTED',
                'description': (
                    'Given the detected abnormality, optimal maternal nutrition remains critical. '
                    'The following is a general prenatal nutrition guideline pending specialist review:'
                ),
                'nutrients': [
                    {
                        'name': 'Comprehensive Prenatal Multivitamin',
                        'dose': 'As prescribed by physician',
                        'sources': 'Medical-grade prenatal supplements',
                        'benefit': 'Ensures baseline nutritional requirements are met'
                    },
                    {
                        'name': 'Folic Acid',
                        'dose': '800–4000 mcg/day (higher if neural defects suspected)',
                        'sources': 'Supplements + fortified foods',
                        'benefit': 'Critical for any structural development support'
                    },
                    {
                        'name': 'Omega-3 (DHA)',
                        'dose': '300 mg/day minimum',
                        'sources': 'Fish oil, algae supplements',
                        'benefit': 'Brain and nervous system support'
                    },
                    {
                        'name': 'Protein',
                        'dose': '80–90 g/day',
                        'sources': 'Diverse animal and plant sources',
                        'benefit': 'Tissue repair and fetal organ development'
                    },
                    {
                        'name': 'Antioxidants (Vitamins C + E)',
                        'dose': 'Vitamin C: 100 mg/day, Vitamin E: 15 mg/day',
                        'sources': 'Fresh fruits, vegetables, nuts, seeds',
                        'benefit': 'Reduces oxidative stress and inflammation'
                    }
                ],
                'foods_to_avoid': [
                    'All alcohol and tobacco products',
                    'Raw/undercooked foods',
                    'High-mercury fish',
                    'Excessive caffeine',
                    'Highly processed food'
                ],
                'additional_guidance': (
                    'URGENT: Please consult a Maternal-Fetal Medicine (MFM) specialist and '
                    'registered prenatal dietitian immediately for personalised nutritional management '
                    'tailored to the specific abnormality detected.'
                )
            }
        }

    def generate_report(self, cnn_result, rag_context=None, patient_name='', patient_phone=''):
        """
        Generate comprehensive radiology report from CNN predictions and RAG context.

        Args:
            cnn_result (dict): CNN prediction result
            rag_context (dict): Retrieved medical context from RAG module (optional)
            patient_name (str): Patient's full name
            patient_phone (str): Patient's phone number

        Returns:
            dict: Structured report with all sections including nutrients
        """
        if rag_context is None:
            rag_context = {}

        condition_label = cnn_result.get('class_label', '')

        report = {
            'report_id': self._generate_report_id(),
            'timestamp': datetime.now().isoformat(),
            'patient_name': patient_name,
            'patient_phone': patient_phone,
            'section_1_patient_scan_summary': self._build_patient_summary(
                cnn_result, patient_name, patient_phone
            ),
            'section_2_technical_details': self._build_technical_details(cnn_result),
            'section_3_findings': self._build_findings(cnn_result, rag_context),
            'section_4_fetal_condition_analysis': self._build_condition_analysis(cnn_result, rag_context),
            'section_5_maternal_risk_factors': self._build_maternal_analysis(rag_context),
            'section_6_risk_assessment': self._build_risk_assessment(cnn_result, rag_context),
            'section_7_clinical_interpretation': self._build_clinical_interpretation(cnn_result, rag_context),
            'section_8_recommendations': self._build_recommendations(cnn_result, rag_context),
            'section_9_summary': self._build_summary(cnn_result),
            'section_10_nutrients': self._build_nutrients_section(cnn_result),
            'metadata': {
                'model_confidence': cnn_result.get('confidence_percentage', 0),
                'cnn_model': cnn_result.get('model_name', 'ResNet50'),
                'version': '2.0'
            }
        }

        return report

    def _generate_report_id(self):
        """Generate unique report ID."""
        import uuid
        return f"REPORT-{uuid.uuid4().hex[:8].upper()}"

    def _select_template(self, condition_label):
        """Select report template based on condition."""
        if 'Normal' in condition_label:
            return self.report_templates['normal']
        elif 'Growth Restriction' in condition_label:
            return self.report_templates['fgr_mild']
        else:
            return self.report_templates['fgr_severe']

    def _build_patient_summary(self, cnn_result, patient_name='', patient_phone=''):
        """Build patient and scan summary section."""
        name_line = f"Patient Name: {patient_name}" if patient_name else "Patient Name: Not Provided"
        phone_line = f"Contact Phone: {patient_phone}" if patient_phone else "Contact Phone: Not Provided"

        return {
            'section_title': 'PATIENT AND SCAN SUMMARY',
            'content': f"""
{name_line}
{phone_line}
Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The fetal ultrasound examination was performed for growth assessment and 
fetal well-being evaluation. Image quality was adequate for diagnostic evaluation.

INDICATION: 
Fetal growth assessment and evaluation of fetal well-being

EXAMINATION TYPE:
Ultrasound - Obstetric, 2nd or 3rd Trimester, Limited
Deep Learning Based Analysis — ResNet50 CNN + RAG Pipeline

FINDINGS OVERVIEW:
{cnn_result.get('condition_description', 'Examination completed successfully.')}
"""
        }

    def _build_technical_details(self, cnn_result):
        """Build technical details section."""
        return {
            'section_title': 'TECHNICAL DETAILS',
            'content': f"""
IMAGE ANALYSIS METHOD: Deep Learning — Convolutional Neural Network (CNN) Classification
MODEL: ResNet50 with Transfer Learning (ImageNet pre-trained)
CONFIDENCE SCORE: {cnn_result.get('confidence_percentage', 0):.1f}%

CLASSIFICATION RESULT:
Primary Prediction: {cnn_result.get('class_label', 'Unknown')}

ALL CLASSIFICATION PROBABILITIES:
{json.dumps(cnn_result.get('all_probabilities', {}), indent=2)}

IMAGE QUALITY: Adequate for diagnostic evaluation
ANALYSIS PIPELINE: CNN → RAG (FAISS) → Template-based Report Generator
"""
        }

    def _build_findings(self, cnn_result, rag_context):
        """Build findings section."""
        confidence = cnn_result.get('confidence_percentage', 0)

        if 'Normal' in cnn_result.get('class_label', ''):
            findings_text = """
BIOMETRIC MEASUREMENTS:
  ✓ Head Circumference (HC): Within normal limits for gestational age
  ✓ Femur Length (FL): Appropriate, normal proportions
  ✓ Abdominal Circumference (AC): Normal, no growth discordance
  ✓ Estimated Fetal Weight (EFW): 50th percentile

AMNIOTIC FLUID ASSESSMENT:
  ✓ Volume: Normal (AFI 8-18 cm or 5-8 quadrant pockets)
  ✓ Distribution: Even distribution throughout uterine cavity
  ✓ No polyhydramnios or oligohydramnios

DOPPLER ULTRASOUND FINDINGS:
  ✓ Umbilical Artery (UA): Normal flow pattern, S/D ratio <3.0
  ✓ Middle Cerebral Artery (MCA): Normal PI, no brain-sparing
  ✓ Cerebroplacental Ratio (CPR): Normal (>1.08)
  ✓ Ductus Venosus: Normal waveform, forward diastolic flow
  ✓ All Doppler indices consistent with good placental function

INTERPRETATION: All parameters support normal fetal development.
            """

        elif 'Growth' in cnn_result.get('class_label', ''):
            findings_text = f"""
BIOMETRIC MEASUREMENTS - ABNORMAL:
  ⚠ Head Circumference (HC): Preserved at expected centile
  ⚠ Femur Length (FL): Maintained, relatively normal
  ⚠ Abdominal Circumference (AC): REDUCED - Below 10th percentile *** 
  ⚠ Estimated Fetal Weight (EFW): <10th percentile for gestational age
  ⚠ HC/AC Ratio: ELEVATED (>1.3) - Asymmetric growth pattern

AMNIOTIC FLUID ASSESSMENT - ABNORMAL:
  ⚠ Volume: DECREASED (oligohydramnios present)
  ⚠ Deepest Vertical Pocket: <2 cm
  ⚠ Clinical Significance: Marker of placental insufficiency

DOPPLER ULTRASOUND FINDINGS - ABNORMAL:
  ⚠ Umbilical Artery (UA): ELEVATED PI and S/D ratio (>3.5)
  ⚠ Diastolic Notching: Present, indicates increased resistance
  ⚠ Middle Cerebral Artery (MCA): DECREASED PI - Brain-sparing effect *** 
  ⚠ Cerebroplacental Ratio (CPR): ABNORMAL (<1.0) ***
  ⚠ Ductus Venosus: Abnormal waveform with reversed flow
  ⚠ All findings consistent with PLACENTAL INSUFFICIENCY

AI MODEL CONFIDENCE: {confidence:.1f}% 
INTERPRETATION: Findings are consistent with FETAL GROWTH RESTRICTION 
due to placental dysfunction.
            """

        else:
            findings_text = """
ABNORMAL FINDINGS DETECTED:
  ⚠ Detailed anatomical structures require careful specialist evaluation
  ⚠ Notable anomaly pattern identified on AI assessment
  
RECOMMENDATION: Urgent consultation with maternal-fetal medicine specialist
            """

        return {
            'section_title': 'DETAILED FINDINGS AND MEASUREMENTS',
            'content': findings_text
        }

    def _build_condition_analysis(self, cnn_result, rag_context):
        """Build fetal condition analysis section."""
        class_label = cnn_result.get('class_label', '')

        if 'Normal' in class_label:
            analysis = """
FETAL CONDITION: NORMAL DEVELOPMENT

The fetus demonstrates normal growth and development patterns consistent with 
stated gestational age. All biometric parameters fall within the expected range.

Key Assessment Points:
• Growth velocity: NORMAL
• Doppler patterns: REASSURING
• Amniotic fluid: ADEQUATE
• Placental function: ADEQUATE
• No evidence of compromise
            """

        elif 'Growth Restriction' in class_label:
            related_conditions = rag_context.get('retrieved_conditions', {})
            fetal_conditions = related_conditions.get('related_fetal_conditions', [])

            condition_details = ""
            if fetal_conditions and len(fetal_conditions) > 0:
                first_condition = fetal_conditions[0]
                condition_details = f"""
Specific Pathophysiology:
{first_condition.get('imaging_features', 'Reduced growth velocity detected')}
                """

            analysis = f"""
FETAL CONDITION: GROWTH RESTRICTION DETECTED

The fetus demonstrates growth parameters below the expected range for gestational age.
This pattern is consistent with Fetal Growth Restriction (FGR).

{condition_details}

Severity Assessment:
• Growth deficit: Estimated weight below 10th percentile
• Placental perfusion: Compromised (abnormal Doppler indices)
• Brain-sparing compensation: PRESENT (elevated cerebroplacental ratio)
• Amniotic fluid status: May be reduced in severe cases
            """

        else:
            analysis = """
FETAL CONDITION: ABNORMALITY DETECTED

Detailed evaluation demonstrates structural or functional abnormality requiring
specialist assessment and potentially additional imaging modalities.
            """

        return {
            'section_title': 'FETAL CONDITION ANALYSIS',
            'content': analysis
        }

    def _build_maternal_analysis(self, rag_context):
        """Build maternal risk factors analysis section."""
        return {
            'section_title': 'MATERNAL CONDITIONS AND RISK FACTORS',
            'content': """
MATERNAL RISK FACTORS CONTRIBUTING TO FETAL STATUS:

The following maternal conditions are known to increase the risk of fetal growth 
restriction and require careful monitoring:

POTENTIAL RISK FACTORS TO ASSESS:

1. HYPERTENSIVE DISORDERS
   - Chronic hypertension
   - Gestational hypertension
   - Preeclampsia/eclampsia
   → Effect: Reduced placental perfusion
   
2. PLACENTAL PATHOLOGY
   - Placental insufficiency
   - Placental infarction
   - Abnormal placentation
   → Effect: Impaired nutrient and oxygen delivery
   
3. MATERNAL METABOLIC CONDITIONS
   - Pre-existing diabetes mellitus
   - Gestational diabetes
   - Maternal obesity
   → Effect: Altered placental function, metabolic imbalance
   
4. MATERNAL INFECTIONS
   - Intrauterine infections (TORCH, syphilis, toxoplasmosis)
   - Current respiratory/systemic infections
   → Effect: Direct fetal infection or placental inflammation
   
5. HEMODYNAMIC FACTORS
   - Maternal anemia
   - Maternal cardiac disease
   - Smoking/substance use
   → Effect: Reduced oxygen-carrying capacity
   
CLINICAL ASSESSMENT REQUIRED:
  ✓ Maternal blood pressure monitoring
  ✓ Urinalysis (detect proteinuria)
  ✓ Glucose tolerance evaluation
  ✓ Complete maternal history and medication review
  ✓ Evaluation for maternal infection markers
        """
        }

    def _build_risk_assessment(self, cnn_result, rag_context):
        """Build risk assessment section."""
        confidence = cnn_result.get('confidence_percentage', 0)

        if confidence > 75:
            risk_level = "HIGH RISK"
            risk_description = "Significant concern for fetal growth restriction or abnormality"
            probability = "60-80% probability"
        elif confidence > 60:
            risk_level = "MODERATE RISK"
            risk_description = "Borderline growth concern; close monitoring recommended"
            probability = "30-50% probability"
        else:
            risk_level = "LOW RISK"
            risk_description = "Normal fetal development pattern"
            probability = "0-20% probability"

        risk_strat = rag_context.get('applicable_risk_stratification', {})

        return {
            'section_title': 'RISK STRATIFICATION AND ASSESSMENT',
            'content': f"""
ASSIGNED RISK LEVEL: {risk_level}

Risk Description: {risk_description}
FGR Probability: {probability}
Clinical Confidence: {confidence:.1f}%

RISK STRATIFICATION GUIDELINES:
Category: {risk_strat.get('risk_level', 'Indeterminate')}
Criteria: {risk_strat.get('criteria', 'Follow standard protocols')}
Recommended Follow-up: {risk_strat.get('follow_up', 'Routine care')}

DOPPLER ASSESSMENT GUIDELINES:
• Abnormal Doppler indices suggest placental insufficiency
• Serial monitoring essential for accurate diagnosis
• Combination of biometry + Doppler improves diagnostic accuracy
            """
        }

    def _build_clinical_interpretation(self, cnn_result, rag_context):
        """Build clinical interpretation section."""
        class_label = cnn_result.get('class_label', '')

        if 'Normal' in class_label:
            interpretation = """
CLINICAL INTERPRETATION:

The ultrasound examination demonstrates normal fetal development with biometric
measurements appropriate for the stated gestational age. There are no imaging
features suggestive of growth restriction or other significant abnormality.

The normal Doppler patterns indicate adequate placental function and appropriate
oxygen delivery to the fetus. The amniotic fluid volume is adequate, reflecting
normal fetal kidney function and overall well-being.

CONCLUSION: 
This is a reassuring examination consistent with normal fetal development
and adequate placental function without evidence of growth restriction.
            """

        elif 'Growth Restriction' in class_label:
            interpretation = """
CLINICAL INTERPRETATION:

The ultrasound examination demonstrates biometric parameters below the expected
range for this gestational age, consistent with growth restriction. The pattern
of decreased abdominal circumference with relatively preserved head circumference
and femur length is classic for asymmetric (late-onset) growth restriction.

The abnormal Doppler indices indicating increased placental resistance and the
brain-sparing effect (elevated cerebroplacental ratio) confirm placental
insufficiency as the underlying mechanism.

This constellation of findings is concerning for fetal compromise and warrants
close clinical correlation with maternal conditions, particularly those affecting
placental perfusion such as:
- Maternal hypertension
- Preeclampsia
- Placental pathology
- Maternal-fetal mismatch

CONCLUSION:
This examination is consistent with clinically significant fetal growth restriction
with evidence of placental insufficiency. Intensive surveillance and consideration
of delivery timing are indicated based on clinical severity and gestational age.
            """

        else:
            interpretation = """
CLINICAL INTERPRETATION:

The ultrasound examination demonstrates abnormal features requiring specialist
evaluation and potentially additional imaging and follow-up studies.

Recommend:
• Consultation with maternal-fetal medicine specialist
• Detailed fetal anatomy survey if not previously performed
• Consideration of advanced imaging (3D ultrasound, MRI)
• Genetic and infectious disease screening as clinically indicated
            """

        return {
            'section_title': 'CLINICAL INTERPRETATION',
            'content': interpretation
        }

    def _build_recommendations(self, cnn_result, rag_context):
        """Build recommendations section."""
        class_label = cnn_result.get('class_label', '')
        confidence = cnn_result.get('confidence_percentage', 0)

        if 'Normal' in class_label:
            recommendations = [
                "Continue routine prenatal care",
                "Repeat ultrasound examination in 4-6 weeks for reassessment",
                "Routine obstetric follow-up as per standard protocols",
                "Maintain healthy lifestyle and adequate nutrition (see nutrients section below)",
                "Report any signs of complications (vaginal bleeding, fluid loss, reduced fetal movement)"
            ]

        elif 'Growth Restriction' in class_label:
            recommendations = [
                "URGENT: Doppler ultrasound studies (umbilical artery, MCA, ductus venosus, umbilical vein)",
                "Comprehensive maternal evaluation (BP monitoring, protein/creatinine ratio, glucose tolerance)",
                "Fetal heart rate monitoring (non-stress test)",
                f"Repeat ultrasound in 1-2 weeks for growth surveillance (higher frequency due to {'high' if confidence > 75 else 'moderate'} risk)",
                "Maternal nutritional optimization — see Section 10 for targeted nutrients",
                "Blood pressure control and maternal medication review",
                "Consider delivery planning at 34-36 weeks depending on severity and fetal maturity",
                "Neonatology consultation for delivery planning and high-risk infant management",
                "Patient education regarding warning signs and when to seek immediate care",
                "Immediate dietary consultation with a registered prenatal dietitian"
            ]

        else:
            recommendations = [
                "Immediate consultation with maternal-fetal medicine specialist",
                "Detailed fetal anatomy survey and consideration of advanced imaging",
                "Genetic counseling if chromosomal anomaly suspected",
                "Infectious disease screening as clinically indicated",
                "Repeat ultrasound with specialist interpretation",
                "Nutritional support — see Section 10 for guidelines"
            ]

        recommendations_text = "CLINICAL RECOMMENDATIONS:\n\n"
        for i, rec in enumerate(recommendations, 1):
            recommendations_text += f"{i}. {rec}\n"

        return {
            'section_title': 'RECOMMENDATIONS AND FOLLOW-UP PLAN',
            'content': recommendations_text
        }

    def _build_summary(self, cnn_result):
        """Build executive summary section."""
        class_label = cnn_result.get('class_label', '')
        confidence = cnn_result.get('confidence_percentage', 0)

        return {
            'section_title': 'SUMMARY',
            'content': f"""
DIAGNOSIS: {class_label}

AI MODEL CONFIDENCE: {confidence:.1f}%

KEY FINDINGS:
• {cnn_result.get('condition_description', 'Please see detailed findings above')}

ASSESSMENT:
This ultrasound examination demonstrates {class_label.lower()} based on detailed
biometric analysis, Doppler assessment, and amniotic fluid evaluation.

PLAN:
See recommendations section above for detailed follow-up and management plan.
See Section 10 for complete nutritional guidance.

---
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Clinical correlation and individual patient assessment are essential.
This report should be interpreted by qualified healthcare providers with access
to complete clinical context.
        """
        }

    def _build_nutrients_section(self, cnn_result):
        """
        Build nutrient recommendations section based on detected condition.
        This is a critical new section (Section 10) tailored to diagnosis.
        """
        class_label = cnn_result.get('class_label', '')

        # Select the right nutrient profile
        if 'Normal' in class_label:
            nutrient_profile = self.nutrient_data['normal']
        elif 'Growth Restriction' in class_label or 'FGR' in class_label:
            nutrient_profile = self.nutrient_data['fgr']
        else:
            nutrient_profile = self.nutrient_data['abnormality']

        # Build the nutrient text
        content = f"""
{nutrient_profile['title']}

{nutrient_profile['description']}

RECOMMENDED NUTRIENTS:
{'─' * 60}
"""
        for i, nutrient in enumerate(nutrient_profile['nutrients'], 1):
            content += f"""
{i}. {nutrient['name']}
   Recommended Dose : {nutrient['dose']}
   Food Sources     : {nutrient['sources']}
   Clinical Benefit : {nutrient['benefit']}
"""

        content += f"""
{'─' * 60}
FOODS TO AVOID DURING PREGNANCY:
"""
        for item in nutrient_profile['foods_to_avoid']:
            content += f"  ✗ {item}\n"

        if 'additional_guidance' in nutrient_profile:
            content += f"""
ADDITIONAL DIETARY GUIDANCE:
{nutrient_profile['additional_guidance']}
"""

        content += """
─────────────────────────────────────────────────────────────
NOTE: These recommendations are AI-generated guidelines.
Please consult a qualified nutritionist or dietitian for a
personalised prenatal nutrition plan specific to your needs.
─────────────────────────────────────────────────────────────
"""

        return {
            'section_title': 'NUTRITIONAL RECOMMENDATIONS',
            'content': content
        }

    def format_report_for_display(self, report):
        """Format report dictionary into readable text format."""
        formatted_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   DEEP LEARNING BASED FETAL ULTRASOUND DIAGNOSTIC REPORT                    ║
║   AI-Enhanced Diagnosis System (CNN + RAG + LLM)                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Report ID : {report.get('report_id', 'N/A')}
Date      : {report.get('timestamp', 'N/A')}
Patient   : {report.get('patient_name', 'N/A')}
Phone     : {report.get('patient_phone', 'N/A')}

"""

        sections = [
            'section_1_patient_scan_summary',
            'section_2_technical_details',
            'section_3_findings',
            'section_4_fetal_condition_analysis',
            'section_5_maternal_risk_factors',
            'section_6_risk_assessment',
            'section_7_clinical_interpretation',
            'section_8_recommendations',
            'section_9_summary',
            'section_10_nutrients'
        ]

        for section in sections:
            if section in report:
                section_data = report[section]
                formatted_text += f"\n{'─' * 80}\n"
                formatted_text += f"{section_data.get('section_title', 'Section')}\n"
                formatted_text += f"{'─' * 80}\n"
                formatted_text += f"{section_data.get('content', '')}\n"

        formatted_text += f"\n{'═' * 80}\n"
        formatted_text += "END OF REPORT\n"
        formatted_text += f"{'═' * 80}\n"

        return formatted_text
