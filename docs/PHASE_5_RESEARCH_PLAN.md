# Phase 5 Research Plan: Advanced Classification System

## 🔬 **Research Overview**

**Status**: CONDITIONAL - Only implement if Phase 4 evaluation shows clear need

**Timeline**: 8-10 weeks (if pursued)

**Objective**: Develop sophisticated interdisciplinary classification system using ML to enhance cross-domain knowledge discovery

---

## 🎯 **Research Goals**

### **Primary Objective**
Create a three-tier classification system that enables interdisciplinary connections between technical (CS) and humanities documents for enhanced graph-based retrieval.

### **Key Research Questions**
1. Can Annif-bootstrapped training data produce high-quality classification models?
2. Do ML-generated bridge keywords improve cross-domain discovery over simple approaches?
3. What's the ROI of sophisticated classification vs. organic graph enhancement?
4. How can we measure meaningful interdisciplinary connections?

---

## 🏗️ **System Architecture**

### **Three-Tier Classification System**

#### **Tier 1: Domain-Specific Classification**
- **Technical Documents**: ACM Computing Classification System (CCS)
- **Humanities/Social Sciences**: HASSET (Humanities and Social Science Electronic Thesaurus)
- **Both are freely available** and provide deep, expert-validated taxonomies

#### **Tier 2: Document Type Classification**
- Academic Paper (journal article, conference paper, preprint)
- Code Artifact (repository, notebook, library)
- Thesis (PhD dissertation, Masters thesis)
- Dataset (research data, corpus)
- Documentation (technical docs, tutorials)

#### **Tier 3: Dynamic Bridge Keywords**
- **LLM-generated** abstract concepts that span disciplines
- 10 keywords per document focusing on universal concepts
- Examples: NETWORK, HIERARCHY, FLOW, PATTERN, BOUNDARY, EXCHANGE, STRUCTURE

---

## 📊 **Research Implementation Pipeline**

### **Phase 5.1: Data Collection & Annif Bootstrapping** (3 weeks)

#### **Corpus Collection Strategy**
```python
class DataCollectionPipeline:
    def collect_diverse_corpus(self):
        """
        Target: 100k+ high-quality documents across domains
        
        Sources:
        - ArXiv papers (CS domain): 30k papers
        - PubMed Central (interdisciplinary): 25k papers  
        - GitHub repos with substantial READMEs: 20k repos
        - Open access humanities papers: 15k papers
        - Thesis repositories: 10k theses
        """
        
    def ensure_quality_and_balance(self):
        """
        Quality criteria:
        - Minimum 1000 words of meaningful content
        - Clear domain classification possible
        - High-confidence Annif classifications
        - Balanced across LCC main classes
        """
```

#### **Annif Bootstrapping**
```python
class AnnifBootstrapper:
    def __init__(self):
        self.annif_projects = [
            'lcc-en',      # Library of Congress Classification
            'lcsh-en',     # Library of Congress Subject Headings
            'yso-en'       # YSO General Finnish Ontology (multilingual)
        ]
    
    def generate_training_labels(self, documents):
        """
        Use Annif to automatically classify documents.
        
        Quality filtering:
        - Only keep classifications with confidence > 0.7
        - Require agreement between multiple Annif models
        - Manual spot-checking of high-stakes classifications
        """
        
        high_quality_labels = []
        
        for doc in documents:
            lcc_result = self.annif.suggest('lcc-en', doc.text)
            lcsh_result = self.annif.suggest('lcsh-en', doc.text)
            
            if (lcc_result[0].score > 0.7 and 
                len(lcsh_result) >= 3 and
                lcsh_result[0].score > 0.6):
                
                high_quality_labels.append({
                    'text': doc.text,
                    'lcc': lcc_result[0].label,
                    'lcsh': [s.label for s in lcsh_result[:5]],
                    'confidence': (lcc_result[0].score + lcsh_result[0].score) / 2
                })
        
        return high_quality_labels
```

### **Phase 5.2: Model Training & Validation** (4 weeks)

#### **Training Architecture**
```python
class ClassificationModelTraining:
    def __init__(self):
        self.base_model = "Qwen/Qwen2.5-7B-Instruct"
        self.training_strategy = "LoRA"  # Parameter-efficient fine-tuning
        
    def create_training_dataset(self, annif_labels):
        """
        Convert Annif classifications to instruction-tuning format.
        
        Format:
        {
            "instruction": "Classify this document using Library of Congress Classification and extract interdisciplinary bridge keywords",
            "input": document_text[:2000],
            "output": {
                "lcc": "QA76.87",
                "lcc_label": "Machine learning", 
                "bridge_keywords": ["NETWORK", "PATTERN", "LEARNING", "SYSTEM", "OPTIMIZATION"]
            }
        }
        """
        
    def train_with_validation(self):
        """
        Training configuration:
        - 90/10 train/validation split
        - Multi-task learning (classification + keyword extraction)
        - Validation on held-out interdisciplinary papers
        - Early stopping based on validation performance
        """
```

#### **Bridge Keyword Research**
```python
class BridgeKeywordExtractor:
    def __init__(self):
        self.keyword_prompt = """
        Extract 10 broad, high-level keywords from this document that represent 
        abstract concepts that could appear in multiple disciplines.

        Good keywords are single words representing universal concepts:
        NETWORK, HIERARCHY, FLOW, PATTERN, BOUNDARY, EXCHANGE, STRUCTURE, 
        PROCESS, EMERGENCE, CONTROL, SYSTEM, COMMUNICATION, ORGANIZATION

        Focus on concepts that could meaningfully connect documents from 
        different academic domains. Avoid domain-specific jargon.

        Document: {text}

        Bridge Keywords (JSON list):
        """
    
    def validate_bridge_quality(self, keywords, document_domain):
        """
        Research question: Do these keywords actually create meaningful 
        connections between documents from different domains?
        
        Validation approaches:
        - Expert review of keyword quality
        - Measure cross-domain retrieval improvement
        - Analyze if keywords connect semantically related concepts
        """
```

### **Phase 5.3: Integration & Evaluation** (3 weeks)

#### **Graph Enhancement Integration**
```python
class AdvancedGraphEnhancer:
    def __init__(self, trained_model):
        self.classifier = trained_model
        self.graph_updater = GraphUpdater()
        
    def enhance_existing_graph(self, graph, documents):
        """
        Apply trained classification to enhance existing directory-based graph.
        
        Enhancement strategies:
        - Add domain classification edges (weight 1.0)
        - Add document type edges (weight 0.7)  
        - Add bridge keyword edges (weight 0.5)
        - Calculate path strengths for PathRAG
        """
        
        for doc in documents:
            classification = self.classifier.classify(doc)
            
            # Add domain-specific edges
            domain_nodes = self._get_domain_nodes(classification.lcc)
            self.graph_updater.add_classification_edges(doc.id, domain_nodes, 1.0)
            
            # Add bridge keyword edges
            keyword_nodes = self._get_keyword_nodes(classification.bridge_keywords)
            self.graph_updater.add_keyword_edges(doc.id, keyword_nodes, 0.5)
```

#### **Evaluation Framework**
```python
class Phase5EvaluationSuite:
    def compare_against_baseline(self, simple_graph, enhanced_graph):
        """
        Rigorous comparison of simple vs. advanced classification.
        
        Metrics:
        - Cross-domain retrieval improvement
        - Query satisfaction scores
        - Discovery of unexpected connections
        - Maintenance overhead comparison
        - User preference studies
        """
        
        return {
            'retrieval_improvement': self._measure_retrieval_quality(),
            'discovery_enhancement': self._measure_serendipitous_discovery(),
            'user_satisfaction': self._conduct_user_studies(),
            'maintenance_cost': self._calculate_total_overhead(),
            'roi_analysis': self._compute_return_on_investment()
        }
```

---

## 📋 **Research Deliverables**

### **Technical Deliverables**
- [ ] Annif-bootstrapped training dataset (100k+ high-quality classifications)
- [ ] Fine-tuned 7B classification model with multi-task capabilities
- [ ] Bridge keyword extraction system with quality validation
- [ ] Integration layer for existing HADES architecture
- [ ] Comprehensive evaluation comparing simple vs. advanced approaches

### **Research Publications**
- [ ] Paper on Annif bootstrapping for classification training data
- [ ] Study on bridge keywords for interdisciplinary knowledge discovery
- [ ] Evaluation of ML vs. simple approaches in RAG systems
- [ ] Case studies of cross-domain discovery improvements

### **Documentation**
- [ ] Complete implementation guide for classification system
- [ ] Training data generation methodology
- [ ] Model performance benchmarks and comparisons
- [ ] Integration guide for existing HADES installations

---

## 🎯 **Success Criteria**

### **Minimum Viable Outcomes**
- [ ] Classification accuracy > 85% on domain-specific taxonomies
- [ ] Bridge keywords create measurable cross-domain connections
- [ ] System integrates seamlessly with existing HADES architecture
- [ ] Clear evidence of improvement over simple approaches

### **Stretch Goals**
- [ ] Classification accuracy > 90% across all domains
- [ ] 30%+ improvement in cross-domain retrieval quality
- [ ] User preference for advanced system in blind studies
- [ ] Published research contributions to the field

### **Failure Criteria** (when to abort)
- [ ] Classification accuracy < 70% after substantial training effort
- [ ] No measurable improvement over simple directory structure
- [ ] Excessive maintenance overhead outweighs benefits
- [ ] User studies show preference for simpler system

---

## 💰 **Resource Requirements**

### **Computational Resources**
- GPU cluster for 7B model training (estimated 200-400 GPU hours)
- Storage for 100k+ document corpus (~500GB)
- Inference infrastructure for real-time classification

### **Data Resources**
- Access to diverse academic document repositories
- Annif server setup with multiple classification models
- Quality validation datasets for evaluation

### **Human Resources**
- ML researcher/engineer for model development
- Domain experts for classification quality validation
- User experience researcher for evaluation studies

---

## 🔄 **Integration with Main Development**

### **Non-Blocking Development**
- Phase 5 research proceeds independently of Phase 2-4 main development
- Placeholder architecture in Phase 2 ready for Phase 5 integration
- Simple classification provides fallback if research doesn't meet goals

### **Decision Points**
1. **After Phase 4**: Evaluate if Phase 5 research is needed based on user satisfaction metrics
2. **Mid-Phase 5.2**: Evaluate training progress and decide whether to continue
3. **End of Phase 5.3**: Final ROI analysis and decision on production deployment

### **Risk Mitigation**
- Core HADES functionality never depends on Phase 5 research outcomes
- Simple classification system remains as fallback
- Research can be discontinued without affecting main product

---

## 📈 **Expected Research Impact**

### **If Successful**
- Significant advancement in interdisciplinary knowledge discovery
- Novel approach to ML-enhanced RAG systems
- Contributions to academic classification and information retrieval
- Enhanced user experience for cross-domain research

### **If Unsuccessful**
- Validation that simple approaches are sufficient for most use cases
- Research insights into limitations of current ML classification approaches
- Preserved development velocity on core HADES functionality
- Clear data on ROI of sophisticated classification systems

This research plan ensures rigorous investigation of advanced classification while maintaining HADES development momentum and preserving fallback options.