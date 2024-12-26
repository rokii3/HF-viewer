import streamlit as st

# Configure the page before anything else
st.set_page_config(
    page_title="HF Dataset Viewer",
    page_icon="🤗",
    layout="wide"
)

# Now import other dependencies
import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import list_datasets
from datasets import get_dataset_config_names
from collections import defaultdict
from typing import Dict, Any, List
import json

class DatasetQualityAnalyzer:
    def __init__(self):
        self.quality_metrics = defaultdict(dict)
    
    def analyze_column(self, column_data: list, column_name: str) -> dict:
        """Performs comprehensive quality analysis on a single column."""
        metrics = {}
        
        # Basic completeness metrics
        total_values = len(column_data)
        null_values = sum(1 for x in column_data if x is None or (isinstance(x, str) and not x.strip()))
        metrics['completeness'] = {
            'total_values': total_values,
            'null_count': null_values,
            'completeness_ratio': (total_values - null_values) / total_values if total_values > 0 else 0
        }
        
        # Data type analysis
        type_counts = defaultdict(int)
        for value in column_data:
            type_counts[str(type(value).__name__)] += 1
        metrics['type_consistency'] = {
            'type_distribution': dict(type_counts),
            'primary_type': max(type_counts.items(), key=lambda x: x[1])[0]
        }
        
        # Statistical metrics for numeric data
        numeric_data = [x for x in column_data if isinstance(x, (int, float)) and x is not None]
        if numeric_data:
            metrics['numeric_stats'] = {
                'mean': float(np.mean(numeric_data)),
                'std': float(np.std(numeric_data)),
                'min': float(np.min(numeric_data)),
                'max': float(np.max(numeric_data)),
                'outliers': self._detect_outliers(numeric_data)
            }
        
        # Text analysis for string data
        text_data = [str(x) for x in column_data if isinstance(x, str) and x.strip()]
        if text_data:
            metrics['text_stats'] = {
                'avg_length': np.mean([len(x) for x in text_data]),
                'unique_ratio': len(set(text_data)) / len(text_data) if text_data else 0,
                'character_set': list(set(''.join(text_data[:1000])))
            }
        
        return metrics
    
    def _detect_outliers(self, data: list) -> dict:
        """Detects outliers using IQR method."""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        return {
            'count': len(outliers),
            'percentage': len(outliers) / len(data) if data else 0,
            'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
        }
    
    def generate_quality_report(self, dataset) -> dict:
        """Generates a comprehensive quality report for the entire dataset."""
        try:
            # Convert dataset to pandas for easier analysis
            df = dataset.to_pandas()
            
            # Analyze each column
            column_reports = {}
            for column in df.columns:
                column_reports[column] = self.analyze_column(df[column].tolist(), column)
            
            # Overall dataset metrics
            overall_metrics = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'completeness_score': np.mean([
                    col_metrics['completeness']['completeness_ratio']
                    for col_metrics in column_reports.values()
                ])
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(column_reports, overall_metrics)
            
            return {
                'overall_metrics': overall_metrics,
                'column_metrics': column_reports,
                'recommendations': recommendations
            }
        except Exception as e:
            st.error(f"Error generating quality report: {str(e)}")
            return {
                'overall_metrics': {},
                'column_metrics': {},
                'recommendations': [f"Error during analysis: {str(e)}"]
            }
    
    def _generate_recommendations(self, column_metrics: dict, overall_metrics: dict) -> list:
        """Generates actionable recommendations based on the analysis."""
        recommendations = []
        
        for column, metrics in column_metrics.items():
            # Completeness recommendations
            if metrics['completeness']['completeness_ratio'] < 0.95:
                recommendations.append(
                    f"Column '{column}' has {metrics['completeness']['null_count']} missing values. "
                    "Consider imputation or investigating the cause of missing data."
                )
            
            # Type consistency recommendations
            if len(metrics['type_consistency']['type_distribution']) > 1:
                recommendations.append(
                    f"Column '{column}' has mixed types: {metrics['type_consistency']['type_distribution']}. "
                    "Consider standardizing the data type."
                )
            
            # Numeric outlier recommendations
            if 'numeric_stats' in metrics and metrics['numeric_stats']['outliers']['percentage'] > 0.01:
                recommendations.append(
                    f"Column '{column}' has {metrics['numeric_stats']['outliers']['count']} outliers "
                    f"({metrics['numeric_stats']['outliers']['percentage']:.2%}). "
                    "Consider investigating these values."
                )
            
            # Text data recommendations
            if 'text_stats' in metrics:
                if metrics['text_stats']['unique_ratio'] == 1.0:
                    recommendations.append(
                        f"Column '{column}' has all unique values. "
                        "This might be an ID column or need deduplication."
                    )
                elif metrics['text_stats']['unique_ratio'] < 0.01:
                    recommendations.append(
                        f"Column '{column}' has very low cardinality. "
                        "Consider if this could be converted to a categorical variable."
                    )
        
        return recommendations

class DatasetMetadataAnalyzer:
    def __init__(self):
        self.huggingface_base_url = "https://huggingface.co/datasets"
    
    def analyze_dataset_metadata(self, dataset_name, dataset):
        """Analyze dataset metadata and documentation."""
        try:
            # Basic dataset information
            metadata = {
                'dataset_name': dataset_name,
                'size': len(dataset),
                'feature_types': {name: str(feature) for name, feature in dataset.features.items()},
                'documentation_url': f"{self.huggingface_base_url}/{dataset_name}",
            }
            
            # Analyze text features
            text_stats = self._analyze_text_features(dataset)
            if text_stats:
                metadata['text_statistics'] = text_stats
            
            # Get configuration info
            metadata['config'] = self._get_dataset_config(dataset)
            
            return metadata
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_text_features(self, dataset):
        """Analyze text features for token statistics."""
        text_stats = {}
        
        try:
            # Sample the dataset to avoid processing too much data
            sample_size = min(1000, len(dataset))
            sample = dataset.select(range(sample_size))
            
            # Analyze each feature that might contain text
            for feature_name, feature in dataset.features.items():
                if str(feature).startswith(('string', 'Value(dtype=')):
                    stats = self._compute_text_statistics(sample, feature_name)
                    if stats:
                        text_stats[feature_name] = stats
            
            return text_stats
        except Exception as e:
            return {'error': str(e)}
    
    def _compute_text_statistics(self, sample, feature_name):
        """Compute statistics for a text feature."""
        try:
            # Get all non-None values
            texts = [item[feature_name] for item in sample if item[feature_name] is not None]
            
            if not texts:
                return None
                
            # Basic text statistics
            stats = {
                'average_length': sum(len(str(t)) for t in texts) / len(texts),
                'max_length': max(len(str(t)) for t in texts),
                'min_length': min(len(str(t)) for t in texts),
                'sample_size': len(texts),
                'unique_values': len(set(texts))
            }
            
            # Word-level statistics
            word_counts = [len(str(t).split()) for t in texts]
            stats.update({
                'average_words': sum(word_counts) / len(word_counts),
                'max_words': max(word_counts),
                'min_words': min(word_counts)
            })
            
            return stats
        except Exception:
            return None
    
    def _get_dataset_config(self, dataset):
        """Extract dataset configuration information."""
        config = {}
        
        try:
            if hasattr(dataset, 'builder_name'):
                config['builder_name'] = dataset.builder_name
            if hasattr(dataset, 'config_name'):
                config['config_name'] = dataset.config_name
            if hasattr(dataset, 'version'):
                config['version'] = str(dataset.version)
        except:
            pass
        
        return config
    
    def display_metadata(self, metadata):
        """Display dataset metadata in Streamlit."""
        st.subheader("Dataset Metadata")
        
        if 'error' in metadata:
            st.error(f"Error analyzing metadata: {metadata['error']}")
            return

        # Basic information
        col1, col2 = st.columns(2)
        with col1:
            st.write("Basic Information:")
            st.write(f"- Dataset Size: {metadata.get('size', 'N/A'):,} entries")
            st.write(f"- Number of Features: {len(metadata.get('feature_types', {}))}")
        
        with col2:
            st.write("Feature Types:")
            for name, feat_type in metadata.get('feature_types', {}).items():
                st.write(f"- {name}: {feat_type}")
        
        # Text statistics if available
        if 'text_statistics' in metadata:
            st.subheader("Text Statistics")
            for feature_name, stats in metadata['text_statistics'].items():
                with st.expander(f"Statistics for {feature_name}"):
                    if isinstance(stats, dict) and 'error' not in stats:
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Avg Length", f"{stats['average_length']:.1f}")
                        with cols[1]:
                            st.metric("Max Length", stats['max_length'])
                        with cols[2]:
                            st.metric("Unique Values", stats['unique_values'])
                        
                        st.write("Word Statistics:")
                        st.write(f"- Average words per entry: {stats['average_words']:.1f}")
                        st.write(f"- Max words: {stats['max_words']}")
                        st.write(f"- Min words: {stats['min_words']}")
        
        # Configuration information
        if 'config' in metadata and metadata['config']:
            st.subheader("Dataset Configuration")
            for key, value in metadata['config'].items():
                st.write(f"- {key}: {value}")
        
        # Documentation link
        st.markdown(f"[View full documentation on Hugging Face]({metadata['documentation_url']})")

class CategoryExplorer:
    def __init__(self):
        self.category_cache = {}
def identify_category_columns(self, dataset):
        """Identify columns that could be used as categories."""
        try:
            # Get a sample of the dataset
            sample_size = min(1000, len(dataset))
            sample = dataset.select(range(sample_size))
            df = sample.to_pandas()
            
            # Find potential category columns
            category_columns = []
            
            for column in df.columns:
                unique_values = df[column].nunique()
                total_values = len(df)
                
                # Consider a column as categorical if it has a reasonable number of unique values
                if 1 < unique_values < min(50, total_values / 2):
                    category_columns.append({
                        'name': column,
                        'unique_values': unique_values,
                        'example_values': sorted(df[column].unique())[:10]
                    })
            
            return category_columns
        except Exception as e:
            st.error(f"Error identifying categories: {str(e)}")
            return []
    
def display_category_explorer(self, dataset, category_columns):
    """Display interface for exploring dataset by categories."""
    st.subheader("Explore by Category")
    
    if not category_columns:
        st.write("No suitable category columns found in this dataset.")
        return
    
    # Let user select a category column
    selected_category = st.selectbox(
        "Select category to explore:",
        options=[col['name'] for col in category_columns],
        format_func=lambda x: f"{x} ({next(c['unique_values'] for c in category_columns if c['name'] == x)} unique values)"
    )
    
    if selected_category:
        # Get the unique values for the selected category
        selected_info = next(c for c in category_columns if c['name'] == selected_category)
        
        # Show example values
        st.write("Example values:", ", ".join(map(str, selected_info['example_values'])))
        
        # Let user select specific values to explore
        selected_values = st.multiselect(
            f"Select {selected_category} values to explore:",
            options=selected_info['example_values']
        )
        
        if selected_values:
            # Filter and display the data
            try:
                filtered_data = dataset.filter(lambda x: x[selected_category] in selected_values)
                
                # Show sample of filtered data
                st.write(f"Showing sample of filtered data (first 10 entries):")
                df_sample = filtered_data.select(range(min(10, len(filtered_data)))).to_pandas()
                st.dataframe(df_sample)
                
                # Add download button for filtered data
                if st.button("Download filtered data"):
                    df_all = filtered_data.to_pandas()
                    csv = df_all.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{selected_category}_filtered.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error filtering data: {str(e)}")

class EnhancedDatasetExplorer:
    def __init__(self):
        """Initialize the enhanced dataset explorer with additional analysis tools."""
        st.title("Enhanced Hugging Face Dataset Explorer")
        st.write("Browse, analyze, and learn from datasets on the Hugging Face Hub")
        
        # Initialize analysis tools
        self.quality_analyzer = DatasetQualityAnalyzer()
        self.metadata_analyzer = DatasetMetadataAnalyzer()
        self.category_explorer = CategoryExplorer()
        self.dataset_names = self.get_dataset_names()
        
        # Create the main interface
        self.create_tabbed_interface()
    
    @st.cache_data
    def get_dataset_names(_self):
        """Fetch and cache the list of available datasets."""
        try:
            datasets = list(list_datasets(limit=1000))
            return sorted([ds.id for ds in datasets])
        except Exception as e:
            st.error(f"Error fetching dataset list: {str(e)}")
            return []
    
    def load_dataset(self, dataset_name, split="train"):
    
        try:
            with st.spinner(f"Loading dataset '{dataset_name}'..."):
                try:
                    # First check if dataset has multiple configs
                    configs = get_dataset_config_names(dataset_name)
                    
                    if configs:
                        # If we have multiple configs, let user select one
                        selected_config = st.selectbox(
                            "This dataset has multiple configurations. Please select one:",
                            options=configs
                        )
                        
                        if selected_config:
                            try:
                                dataset = load_dataset(dataset_name, selected_config, split=split)
                                return dataset, None
                            except ValueError as e:
                                if "trust_remote_code=True" in str(e):
                                    return self._handle_remote_code(dataset_name, selected_config, split)
                                raise e
                        return None, "config_required"
                    
                    # No configs, try loading directly
                    dataset = load_dataset(dataset_name, split=split)
                    return dataset, None
                    
                except ValueError as e:
                    if "trust_remote_code=True" in str(e):
                        return self._handle_remote_code(dataset_name, None, split)
                    raise e
                    
        except Exception as e:
            st.error(f"Error loading dataset '{dataset_name}': {str(e)}")
            return None, str(e)

    def _handle_remote_code(self, dataset_name, config, split):
        """Handle datasets requiring trust_remote_code=True."""
        st.warning(
            "This dataset contains custom loading code. Running custom code from datasets "
            "could potentially be unsafe. Please review the dataset's code and documentation "
            "before proceeding."
        )
        
        st.markdown(f"[Review dataset code and documentation](https://huggingface.co/datasets/{dataset_name})")
        
        trust_remote = st.checkbox(
            "I have reviewed the dataset code and agree to run it",
            help="This will allow the execution of custom code provided by the dataset"
        )
        
        if trust_remote:
            if config:
                dataset = load_dataset(dataset_name, config, split=split, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
            return dataset, None
        
        return None, "custom_code_required"
    
    def create_tabbed_interface(self):
        """Create a tabbed interface for different features."""
        # Create sidebar for dataset selection
        with st.sidebar:
            st.header("Dataset Selection")
            search_term = st.text_input(
                "Search datasets:",
                help="Filter datasets by name"
            )
            
            filtered_datasets = [
                ds for ds in self.dataset_names 
                if search_term.lower() in ds.lower()
            ] if search_term else self.dataset_names
            
            dataset_name = st.selectbox(
                "Select a dataset:",
                options=filtered_datasets,
                index=None,
                placeholder="Choose a dataset..."
            )
        
        if dataset_name:
            # Load dataset first
            dataset, error_type = self.load_dataset(dataset_name)
            
            if dataset is not None:
                # Create tabs for different features
                tabs = st.tabs([
                    "Dataset Preview", 
                    "Metadata & Statistics",
                    "Category Explorer",
                    "Quality Analysis"
                ])
                
                # Dataset Preview Tab
                with tabs[0]:
                    self.show_dataset_preview(dataset)
                
                # Metadata & Statistics Tab
                with tabs[1]:
                    metadata = self.metadata_analyzer.analyze_dataset_metadata(dataset_name, dataset)
                    self.metadata_analyzer.display_metadata(metadata)
                
                # Category Explorer Tab
                with tabs[2]:
                    category_columns = self.category_explorer.identify_category_columns(dataset)
                    self.category_explorer.display_category_explorer(dataset, category_columns)
                
                # Quality Analysis Tab
                with tabs[3]:
                    self.show_quality_analysis(dataset)
    
    def show_dataset_preview(self, dataset):
        """Show basic dataset preview with enhanced visualization options."""
        st.header("Dataset Preview")
        
        # Basic information
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Total rows: {len(dataset):,}")
            st.write(f"Features: {len(dataset.features)}")
        
        # Sample size selection with dynamic maximum
        sample_size = st.slider(
            "Preview size",
            min_value=1,
            max_value=min(100, len(dataset)),
            value=min(10, len(dataset))
        )
        
        try:
            df = dataset.select(range(sample_size)).to_pandas()
            
            # Column selection for wide datasets
            if len(df.columns) > 10:
                selected_columns = st.multiselect(
                    "Select columns to display:",
                    options=df.columns,
                    default=list(df.columns)[:10]
                )
                df_display = df[selected_columns] if selected_columns else df
            else:
                df_display = df
            
            st.dataframe(df_display)
            
            # Add download option
            csv = df_display.to_csv(index=False)
            st.download_button(
                "Download Preview",
                csv,
                "dataset_preview.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(f"Error displaying preview: {str(e)}")
    
    def show_quality_analysis(self, dataset):
        """Show comprehensive quality analysis of the dataset."""
        st.header("Quality Analysis")
        
        with st.spinner("Analyzing dataset quality..."):
            # Generate quality report
            quality_report = self.quality_analyzer.generate_quality_report(dataset)
            
            # Display overall metrics
            st.subheader("Overall Dataset Quality")
            metrics = quality_report['overall_metrics']
            cols = st.columns(3)
            with cols[0]:
                st.metric("Completeness Score", f"{metrics['completeness_score']:.2%}")
            with cols[1]:
                st.metric("Total Rows", f"{metrics['total_rows']:,}")
            with cols[2]:
                st.metric("Total Columns", metrics['total_columns'])
            
            # Display recommendations
            st.subheader("Recommendations")
            for rec in quality_report['recommendations']:
                st.info(rec)
            
            # Detailed column analysis in expandable sections
            st.subheader("Detailed Column Analysis")
            for col_name, metrics in quality_report['column_metrics'].items():
                with st.expander(f"Column: {col_name}"):
                    self.display_column_metrics(col_name, metrics)
    
    def display_column_metrics(self, column_name, metrics):
        """Display detailed metrics for a single column."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Basic Statistics:")
            st.write(f"- Completeness: {metrics['completeness']['completeness_ratio']:.2%}")
            st.write(f"- Null Count: {metrics['completeness']['null_count']}")
            
            if 'numeric_stats' in metrics:
                st.write("Numeric Statistics:")
                st.write(f"- Mean: {metrics['numeric_stats']['mean']:.2f}")
                st.write(f"- Std Dev: {metrics['numeric_stats']['std']:.2f}")
        
        with col2:
            if 'text_stats' in metrics:
                st.write("Text Statistics:")
                st.write(f"- Average Length: {metrics['text_stats']['avg_length']:.1f}")
                st.write(f"- Unique Ratio: {metrics['text_stats']['unique_ratio']:.2%}")

def main():
    explorer = EnhancedDatasetExplorer()

if __name__ == "__main__":
    main()

