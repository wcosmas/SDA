#!/usr/bin/env python3
"""
Part C: WorkMate Mobile App - Working Code Demonstration
RTV Senior Data Scientist Technical Assessment

This module provides a working demonstration of:
1. Field officer user flow for data input
2. Real-time prediction generation
3. Offline data handling and storage
4. Background synchronization when connected
5. User-friendly interface for field officers

Note: This is a Python simulation of the React Native mobile app
for demonstration purposes.
"""

import sqlite3
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectivityStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    SYNCING = "syncing"

class SyncStatus(Enum):
    PENDING = "pending"
    SYNCING = "syncing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class HouseholdRecord:
    """Household data record structure"""
    household_id: str
    district: str
    cluster: str
    village: str
    household_size: int
    agriculture_land: float
    vsla_profits: float
    business_income: float
    formal_employment: int
    time_to_opd: int
    season1_crops_planted: int
    vehicle_owner: int
    created_at: str
    field_officer_id: str
    sync_status: str = "pending"

@dataclass
class PredictionResult:
    """Prediction result structure"""
    household_id: str
    vulnerability_class: str
    risk_level: str
    confidence: float
    recommendations: List[str]
    prediction_time: str
    model_version: str

class LocalDatabase:
    """Local SQLite database for offline storage"""
    
    def __init__(self, db_path: str = "workmate_local.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize local database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Households table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS households (
                id TEXT PRIMARY KEY,
                district TEXT,
                cluster TEXT,
                village TEXT,
                household_size INTEGER,
                agriculture_land REAL,
                vsla_profits REAL,
                business_income REAL,
                formal_employment INTEGER,
                time_to_opd INTEGER,
                season1_crops_planted INTEGER,
                vehicle_owner INTEGER,
                created_at TEXT,
                field_officer_id TEXT,
                sync_status TEXT DEFAULT 'pending'
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                household_id TEXT PRIMARY KEY,
                vulnerability_class TEXT,
                risk_level TEXT,
                confidence REAL,
                recommendations TEXT,
                prediction_time TEXT,
                model_version TEXT,
                FOREIGN KEY (household_id) REFERENCES households (id)
            )
        ''')
        
        # Sync queue table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_type TEXT,
                record_id TEXT,
                action TEXT,
                data TEXT,
                created_at TEXT,
                retry_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Local database initialized")
    
    def save_household(self, household: HouseholdRecord) -> bool:
        """Save household record to local database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO households 
                (id, district, cluster, village, household_size, agriculture_land,
                 vsla_profits, business_income, formal_employment, time_to_opd,
                 season1_crops_planted, vehicle_owner, created_at, field_officer_id, sync_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                household.household_id, household.district, household.cluster,
                household.village, household.household_size, household.agriculture_land,
                household.vsla_profits, household.business_income, household.formal_employment,
                household.time_to_opd, household.season1_crops_planted, household.vehicle_owner,
                household.created_at, household.field_officer_id, household.sync_status
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving household: {e}")
            return False
    
    def save_prediction(self, prediction: PredictionResult) -> bool:
        """Save prediction result to local database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO predictions
                (household_id, vulnerability_class, risk_level, confidence,
                 recommendations, prediction_time, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.household_id, prediction.vulnerability_class,
                prediction.risk_level, prediction.confidence,
                json.dumps(prediction.recommendations), prediction.prediction_time,
                prediction.model_version
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return False
    
    def get_pending_sync_records(self) -> List[Dict[str, Any]]:
        """Get records pending synchronization"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get households pending sync
        cursor.execute('''
            SELECT * FROM households WHERE sync_status = 'pending'
        ''')
        households = cursor.fetchall()
        
        # Get predictions for households
        household_ids = [h[0] for h in households]
        if household_ids:
            cursor.execute(f'''
                SELECT * FROM predictions 
                WHERE household_id IN ({','.join(['?' for _ in household_ids])})
            ''', household_ids)
            predictions = cursor.fetchall()
        else:
            predictions = []
        
        conn.close()
        
        return {
            'households': households,
            'predictions': predictions
        }
    
    def update_sync_status(self, household_id: str, status: str) -> bool:
        """Update sync status for a household"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE households SET sync_status = ? WHERE id = ?
            ''', (status, household_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error updating sync status: {e}")
            return False

class LocalPredictionEngine:
    """Local prediction engine for offline inference"""
    
    def __init__(self):
        self.model_loaded = False
        self.model_version = "2.0.0-mobile"
        self.load_model()
    
    def load_model(self):
        """Load local ML model (simulated)"""
        try:
            # In a real implementation, this would load TensorFlow Lite model
            logger.info("📱 Loading local prediction model...")
            time.sleep(1)  # Simulate model loading
            self.model_loaded = True
            logger.info("✅ Local model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            self.model_loaded = False
    
    def preprocess_data(self, household_data: Dict[str, Any]) -> Dict[str, float]:
        """Preprocess household data for prediction"""
        # Data cleaning and feature engineering
        processed_data = {
            'household_size': float(household_data.get('household_size', 5)),
            'agriculture_land': float(household_data.get('agriculture_land', 1.5)),
            'vsla_profits': float(household_data.get('vsla_profits', 300)),
            'business_income': float(household_data.get('business_income', 200)),
            'formal_employment': float(household_data.get('formal_employment', 0)),
            'time_to_opd': float(household_data.get('time_to_opd', 60)),
            'season1_crops_planted': float(household_data.get('season1_crops_planted', 3)),
            'vehicle_owner': float(household_data.get('vehicle_owner', 0))
        }
        
        # Add district encoding
        district_mapping = {'Mitooma': 0, 'Kanungu': 1, 'Rubirizi': 2, 'Ntungamo': 3}
        processed_data['district_encoded'] = float(
            district_mapping.get(household_data.get('district', 'Mitooma'), 0)
        )
        
        return processed_data
    
    def predict(self, household_data: Dict[str, Any]) -> PredictionResult:
        """Generate vulnerability prediction"""
        if not self.model_loaded:
            return self._get_fallback_prediction(household_data['household_id'])
        
        try:
            # Preprocess data
            features = self.preprocess_data(household_data)
            
            # Simple rule-based prediction for demo (replaces actual ML model)
            vulnerability_score = self._calculate_vulnerability_score(features)
            
            # Map score to vulnerability class
            if vulnerability_score >= 3:
                vulnerability_class = "Severely Struggling"
                risk_level = "Critical"
                confidence = 0.92
            elif vulnerability_score >= 2:
                vulnerability_class = "Struggling"
                risk_level = "High"
                confidence = 0.88
            elif vulnerability_score >= 1:
                vulnerability_class = "At Risk"
                risk_level = "Medium"
                confidence = 0.85
            else:
                vulnerability_class = "On Track"
                risk_level = "Low"
                confidence = 0.90
            
            # Generate recommendations
            recommendations = self._generate_recommendations(vulnerability_class, features)
            
            prediction = PredictionResult(
                household_id=household_data['household_id'],
                vulnerability_class=vulnerability_class,
                risk_level=risk_level,
                confidence=confidence,
                recommendations=recommendations,
                prediction_time=datetime.now().isoformat(),
                model_version=self.model_version
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return self._get_fallback_prediction(household_data['household_id'])
    
    def _calculate_vulnerability_score(self, features: Dict[str, float]) -> int:
        """Calculate vulnerability score based on features"""
        score = 0
        
        # Household size risk
        if features['household_size'] > 8:
            score += 1
        
        # Agriculture land risk
        if features['agriculture_land'] < 1.0:
            score += 1
        
        # Income risks
        if features['vsla_profits'] < 200:
            score += 1
        
        if features['business_income'] < 150:
            score += 1
        
        # Employment risk
        if features['formal_employment'] == 0:
            score += 0.5
        
        # Healthcare access risk
        if features['time_to_opd'] > 120:
            score += 1
        
        # Agricultural diversity risk
        if features['season1_crops_planted'] < 2:
            score += 0.5
        
        return min(int(score), 4)  # Cap at 4
    
    def _generate_recommendations(self, vulnerability_class: str, features: Dict[str, float]) -> List[str]:
        """Generate intervention recommendations"""
        recommendations = []
        
        if vulnerability_class == "Severely Struggling":
            recommendations.extend([
                "Immediate cash transfer assistance required",
                "Emergency food support enrollment",
                "Healthcare access facilitation",
                "Connect with emergency services"
            ])
        elif vulnerability_class == "Struggling":
            recommendations.extend([
                "Enroll in targeted livelihood programs",
                "Business training and support",
                "Agricultural extension services",
                "VSLA group participation"
            ])
        elif vulnerability_class == "At Risk":
            recommendations.extend([
                "Preventive program enrollment",
                "Savings group formation",
                "Skills training opportunities",
                "Regular monitoring visits"
            ])
        else:
            recommendations.extend([
                "Community program participation",
                "Peer support networks",
                "Economic opportunity sharing"
            ])
        
        # Add specific recommendations based on features
        if features['agriculture_land'] < 1.0:
            recommendations.append("Land productivity improvement training")
        
        if features['season1_crops_planted'] < 2:
            recommendations.append("Crop diversification support")
        
        if features['time_to_opd'] > 120:
            recommendations.append("Healthcare access improvement needed")
        
        return recommendations[:4]  # Limit to top 4 recommendations
    
    def _get_fallback_prediction(self, household_id: str) -> PredictionResult:
        """Provide fallback prediction when model fails"""
        return PredictionResult(
            household_id=household_id,
            vulnerability_class="At Risk",
            risk_level="Medium",
            confidence=0.60,
            recommendations=[
                "Manual assessment recommended",
                "Verify household data",
                "Contact supervisor for guidance"
            ],
            prediction_time=datetime.now().isoformat(),
            model_version="fallback"
        )

class BackgroundSyncManager:
    """Manages background synchronization with backend"""
    
    def __init__(self, db: LocalDatabase, api_base_url: str = "https://api.rtv.org"):
        self.db = db
        self.api_base_url = api_base_url
        self.sync_queue = queue.Queue()
        self.is_syncing = False
        self.connectivity_status = ConnectivityStatus.OFFLINE
        
    def check_connectivity(self) -> bool:
        """Check if device has internet connectivity"""
        try:
            # In real implementation, this would check actual network connectivity
            # For demo, we'll simulate connectivity
            import random
            return random.choice([True, True, True, False])  # 75% chance of connectivity
            
        except:
            return False
    
    def start_background_sync(self):
        """Start background synchronization thread"""
        def sync_worker():
            while True:
                try:
                    if self.check_connectivity():
                        if self.connectivity_status == ConnectivityStatus.OFFLINE:
                            logger.info("📶 Connectivity restored - starting sync")
                        
                        self.connectivity_status = ConnectivityStatus.ONLINE
                        self.sync_pending_data()
                    else:
                        if self.connectivity_status == ConnectivityStatus.ONLINE:
                            logger.info("📵 Connectivity lost - switching to offline mode")
                        
                        self.connectivity_status = ConnectivityStatus.OFFLINE
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in sync worker: {e}")
                    time.sleep(60)  # Wait longer on error
        
        sync_thread = threading.Thread(target=sync_worker, daemon=True)
        sync_thread.start()
        logger.info("🔄 Background sync manager started")
    
    def sync_pending_data(self):
        """Synchronize pending data with backend"""
        if self.is_syncing:
            return
        
        try:
            self.is_syncing = True
            self.connectivity_status = ConnectivityStatus.SYNCING
            
            # Get pending records
            pending_data = self.db.get_pending_sync_records()
            
            if not pending_data['households']:
                return
            
            logger.info(f"🔄 Syncing {len(pending_data['households'])} households...")
            
            # Simulate API calls to backend
            for household_row in pending_data['households']:
                household_id = household_row[0]
                
                # Simulate API call
                success = self._sync_household_to_backend(household_row)
                
                if success:
                    self.db.update_sync_status(household_id, 'completed')
                    logger.debug(f"✅ Synced household {household_id}")
                else:
                    logger.warning(f"❌ Failed to sync household {household_id}")
                
                time.sleep(0.5)  # Throttle requests
            
            logger.info("✅ Data synchronization completed")
            
        except Exception as e:
            logger.error(f"Error during synchronization: {e}")
        finally:
            self.is_syncing = False
            self.connectivity_status = ConnectivityStatus.ONLINE
    
    def _sync_household_to_backend(self, household_row: Tuple) -> bool:
        """Sync individual household to backend (simulated)"""
        try:
            # In real implementation, this would make HTTP POST to backend API
            # For demo, we'll simulate API call with random success/failure
            import random
            
            # Simulate network delay
            time.sleep(random.uniform(0.1, 0.5))
            
            # Simulate 95% success rate
            success = random.random() < 0.95
            
            if success:
                # Simulate successful API response
                response_data = {
                    'status': 'success',
                    'household_id': household_row[0],
                    'message': 'Household data synced successfully'
                }
                return True
            else:
                # Simulate API error
                return False
                
        except Exception as e:
            logger.error(f"Error syncing to backend: {e}")
            return False

class WorkMateApp:
    """Main WorkMate mobile app interface"""
    
    def __init__(self, field_officer_id: str = "FO001"):
        self.field_officer_id = field_officer_id
        self.db = LocalDatabase()
        self.prediction_engine = LocalPredictionEngine()
        self.sync_manager = BackgroundSyncManager(self.db)
        self.current_session = {
            'households_collected': 0,
            'predictions_generated': 0,
            'session_start': datetime.now().isoformat()
        }
        
        # Start background sync
        self.sync_manager.start_background_sync()
        
        logger.info(f"📱 WorkMate App initialized for Field Officer: {field_officer_id}")
    
    def display_home_screen(self):
        """Display main app interface"""
        print("\n" + "="*60)
        print("         🏠 WORKMATE - HOUSEHOLD ASSESSMENT")
        print("="*60)
        print(f"Field Officer: {self.field_officer_id}")
        print(f"Session Start: {self.current_session['session_start'][:19]}")
        print(f"Connectivity: {self.sync_manager.connectivity_status.value.upper()}")
        print("-"*60)
        print("📊 Session Summary:")
        print(f"  Households Assessed: {self.current_session['households_collected']}")
        print(f"  Predictions Generated: {self.current_session['predictions_generated']}")
        print("-"*60)
        print("Available Actions:")
        print("  1. 📝 New Household Assessment")
        print("  2. 📋 View Offline Records")
        print("  3. 🔄 Manual Sync")
        print("  4. ⚙️  Settings")
        print("  5. 📊 Session Report")
        print("  0. 🚪 Exit")
        print("="*60)
    
    def collect_household_data(self) -> Optional[HouseholdRecord]:
        """Collect household data from field officer"""
        print("\n📝 NEW HOUSEHOLD ASSESSMENT")
        print("-"*40)
        
        try:
            # Generate unique household ID
            household_id = f"HH{int(time.time())}"
            
            # Collect basic information
            print("📍 Location Information:")
            district = input("District [Mitooma/Kanungu/Rubirizi/Ntungamo]: ").strip()
            if not district:
                district = "Mitooma"
            
            cluster = input("Cluster: ").strip() or f"CL{household_id[-3:]}"
            village = input("Village: ").strip() or f"VL{household_id[-3:]}"
            
            print("\n👥 Household Information:")
            try:
                household_size = int(input("Household Size [1-20]: ") or "5")
                household_size = max(1, min(20, household_size))
            except ValueError:
                household_size = 5
            
            print("\n🌾 Agricultural Information:")
            try:
                agriculture_land = float(input("Agriculture Land (acres) [0-50]: ") or "1.5")
                agriculture_land = max(0, min(50, agriculture_land))
            except ValueError:
                agriculture_land = 1.5
            
            try:
                season1_crops = int(input("Season 1 Crops Planted [0-10]: ") or "3")
                season1_crops = max(0, min(10, season1_crops))
            except ValueError:
                season1_crops = 3
            
            print("\n💰 Economic Information:")
            try:
                vsla_profits = float(input("VSLA Profits (monthly) [0-10000]: ") or "300")
                vsla_profits = max(0, min(10000, vsla_profits))
            except ValueError:
                vsla_profits = 300
            
            try:
                business_income = float(input("Business Income (monthly) [0-5000]: ") or "200")
                business_income = max(0, min(5000, business_income))
            except ValueError:
                business_income = 200
            
            formal_employment = input("Formal Employment [Y/N]: ").strip().upper()
            formal_employment = 1 if formal_employment == 'Y' else 0
            
            vehicle_owner = input("Vehicle Owner [Y/N]: ").strip().upper()
            vehicle_owner = 1 if vehicle_owner == 'Y' else 0
            
            print("\n🏥 Healthcare Access:")
            try:
                time_to_opd = int(input("Time to OPD (minutes) [0-300]: ") or "60")
                time_to_opd = max(0, min(300, time_to_opd))
            except ValueError:
                time_to_opd = 60
            
            # Create household record
            household = HouseholdRecord(
                household_id=household_id,
                district=district,
                cluster=cluster,
                village=village,
                household_size=household_size,
                agriculture_land=agriculture_land,
                vsla_profits=vsla_profits,
                business_income=business_income,
                formal_employment=formal_employment,
                time_to_opd=time_to_opd,
                season1_crops_planted=season1_crops,
                vehicle_owner=vehicle_owner,
                created_at=datetime.now().isoformat(),
                field_officer_id=self.field_officer_id
            )
            
            print(f"\n✅ Household data collected: {household_id}")
            return household
            
        except KeyboardInterrupt:
            print("\n❌ Data collection cancelled")
            return None
        except Exception as e:
            logger.error(f"Error collecting household data: {e}")
            print(f"❌ Error collecting data: {e}")
            return None
    
    def generate_prediction(self, household: HouseholdRecord) -> Optional[PredictionResult]:
        """Generate vulnerability prediction for household"""
        try:
            print("\n🔮 GENERATING PREDICTION...")
            print("-"*40)
            
            # Convert household to dict for prediction
            household_data = asdict(household)
            
            # Add delay to simulate processing
            print("⏳ Processing household data...")
            time.sleep(1)
            
            print("🧠 Running vulnerability assessment...")
            time.sleep(1)
            
            # Generate prediction
            prediction = self.prediction_engine.predict(household_data)
            
            print("✅ Prediction completed!")
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            print(f"❌ Error generating prediction: {e}")
            return None
    
    def display_prediction_results(self, household: HouseholdRecord, prediction: PredictionResult):
        """Display prediction results to field officer"""
        print("\n" + "="*60)
        print("           🎯 VULNERABILITY ASSESSMENT RESULTS")
        print("="*60)
        
        # Household summary
        print(f"Household ID: {household.household_id}")
        print(f"Location: {household.village}, {household.cluster}, {household.district}")
        print(f"Household Size: {household.household_size}")
        print("-"*60)
        
        # Prediction results
        print("📊 VULNERABILITY ASSESSMENT:")
        print(f"  Status: {prediction.vulnerability_class}")
        print(f"  Risk Level: {prediction.risk_level}")
        print(f"  Confidence: {prediction.confidence:.1%}")
        print(f"  Model Version: {prediction.model_version}")
        
        # Risk level styling
        risk_indicators = {
            "Critical": "🚨 CRITICAL",
            "High": "⚠️  HIGH",
            "Medium": "⚡ MEDIUM",
            "Low": "✅ LOW"
        }
        
        print(f"\n🎯 INTERVENTION PRIORITY: {risk_indicators.get(prediction.risk_level, prediction.risk_level)}")
        
        print("\n📋 RECOMMENDED ACTIONS:")
        for i, recommendation in enumerate(prediction.recommendations, 1):
            print(f"  {i}. {recommendation}")
        
        print("\n⏰ Assessment Time:", prediction.prediction_time[:19])
        print("="*60)
        
        # Ask for confirmation
        confirm = input("\n💾 Save assessment? [Y/n]: ").strip().upper()
        return confirm != 'N'
    
    def save_assessment(self, household: HouseholdRecord, prediction: PredictionResult):
        """Save household assessment and prediction"""
        try:
            # Save to local database
            household_saved = self.db.save_household(household)
            prediction_saved = self.db.save_prediction(prediction)
            
            if household_saved and prediction_saved:
                self.current_session['households_collected'] += 1
                self.current_session['predictions_generated'] += 1
                
                print("✅ Assessment saved successfully")
                
                # Show sync status
                if self.sync_manager.connectivity_status == ConnectivityStatus.ONLINE:
                    print("📶 Will sync to server when connectivity allows")
                else:
                    print("📵 Saved offline - will sync when connected")
                
                return True
            else:
                print("❌ Error saving assessment")
                return False
                
        except Exception as e:
            logger.error(f"Error saving assessment: {e}")
            print(f"❌ Error saving assessment: {e}")
            return False
    
    def view_offline_records(self):
        """Display offline records and sync status"""
        try:
            pending_data = self.db.get_pending_sync_records()
            
            print("\n📋 OFFLINE RECORDS")
            print("-"*40)
            print(f"Pending Sync: {len(pending_data['households'])} households")
            print(f"Connectivity: {self.sync_manager.connectivity_status.value.upper()}")
            
            if pending_data['households']:
                print("\nRecords pending synchronization:")
                for i, household in enumerate(pending_data['households'][:5], 1):
                    household_id, district, village = household[0], household[1], household[3]
                    created_at = household[12][:19]
                    print(f"  {i}. {household_id} - {village}, {district} ({created_at})")
                
                if len(pending_data['households']) > 5:
                    print(f"  ... and {len(pending_data['households']) - 5} more")
            else:
                print("✅ All records synchronized")
            
            print("-"*40)
            
        except Exception as e:
            logger.error(f"Error viewing offline records: {e}")
            print(f"❌ Error viewing records: {e}")
    
    def manual_sync(self):
        """Trigger manual synchronization"""
        print("\n🔄 MANUAL SYNCHRONIZATION")
        print("-"*40)
        
        if self.sync_manager.connectivity_status == ConnectivityStatus.OFFLINE:
            print("📵 No internet connectivity - cannot sync")
            return
        
        if self.sync_manager.is_syncing:
            print("🔄 Sync already in progress...")
            return
        
        print("🔄 Starting manual sync...")
        try:
            self.sync_manager.sync_pending_data()
            print("✅ Manual sync completed")
        except Exception as e:
            logger.error(f"Error in manual sync: {e}")
            print(f"❌ Sync failed: {e}")
    
    def generate_session_report(self):
        """Generate session summary report"""
        print("\n📊 SESSION REPORT")
        print("="*50)
        
        session_duration = datetime.now() - datetime.fromisoformat(self.current_session['session_start'])
        hours, remainder = divmod(session_duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        
        print(f"Field Officer: {self.field_officer_id}")
        print(f"Session Duration: {int(hours)}h {int(minutes)}m")
        print(f"Households Assessed: {self.current_session['households_collected']}")
        print(f"Predictions Generated: {self.current_session['predictions_generated']}")
        
        # Get sync status
        pending_data = self.db.get_pending_sync_records()
        synced_count = self.current_session['households_collected'] - len(pending_data['households'])
        
        print(f"Records Synced: {synced_count}")
        print(f"Pending Sync: {len(pending_data['households'])}")
        print(f"Connectivity Status: {self.sync_manager.connectivity_status.value.upper()}")
        
        if self.current_session['households_collected'] > 0:
            productivity = self.current_session['households_collected'] / max(1, hours + minutes/60)
            print(f"Productivity: {productivity:.1f} assessments/hour")
        
        print("="*50)
    
    def run_demo_workflow(self):
        """Run automated demo workflow"""
        print("\n🎬 RUNNING DEMO WORKFLOW")
        print("="*60)
        
        # Demo household data
        demo_households = [
            {
                'district': 'Mitooma',
                'cluster': 'CL001',
                'village': 'Bukanga',
                'household_size': 8,
                'agriculture_land': 0.5,
                'vsla_profits': 150,
                'business_income': 80,
                'formal_employment': 0,
                'time_to_opd': 120,
                'season1_crops_planted': 1,
                'vehicle_owner': 0
            },
            {
                'district': 'Kanungu',
                'cluster': 'CL002',
                'village': 'Kihihi',
                'household_size': 4,
                'agriculture_land': 2.5,
                'vsla_profits': 600,
                'business_income': 400,
                'formal_employment': 1,
                'time_to_opd': 45,
                'season1_crops_planted': 5,
                'vehicle_owner': 1
            }
        ]
        
        for i, demo_data in enumerate(demo_households, 1):
            print(f"\n📝 Processing Demo Household {i}/2")
            print("-"*40)
            
            # Create household record
            household_id = f"DEMO{int(time.time())}{i}"
            household = HouseholdRecord(
                household_id=household_id,
                created_at=datetime.now().isoformat(),
                field_officer_id=self.field_officer_id,
                **demo_data
            )
            
            print(f"Household: {household.village}, {household.district}")
            print(f"Size: {household.household_size}, Land: {household.agriculture_land} acres")
            
            # Generate prediction
            prediction = self.generate_prediction(household)
            if prediction:
                print(f"Prediction: {prediction.vulnerability_class} ({prediction.confidence:.1%})")
                print(f"Risk Level: {prediction.risk_level}")
                
                # Save assessment
                self.save_assessment(household, prediction)
                
                time.sleep(2)  # Pause between demo households
        
        print("\n✅ Demo workflow completed!")
        self.generate_session_report()
    
    def main_menu(self):
        """Main application menu loop"""
        while True:
            try:
                self.display_home_screen()
                choice = input("\nSelect action [0-5]: ").strip()
                
                if choice == '1':
                    # New household assessment
                    household = self.collect_household_data()
                    if household:
                        prediction = self.generate_prediction(household)
                        if prediction:
                            save_confirmed = self.display_prediction_results(household, prediction)
                            if save_confirmed:
                                self.save_assessment(household, prediction)
                
                elif choice == '2':
                    # View offline records
                    self.view_offline_records()
                    input("\nPress Enter to continue...")
                
                elif choice == '3':
                    # Manual sync
                    self.manual_sync()
                    input("\nPress Enter to continue...")
                
                elif choice == '4':
                    # Settings (placeholder)
                    print("\n⚙️ Settings - Coming Soon")
                    input("\nPress Enter to continue...")
                
                elif choice == '5':
                    # Session report
                    self.generate_session_report()
                    input("\nPress Enter to continue...")
                
                elif choice == '0':
                    # Exit
                    print("\n👋 Thank you for using WorkMate!")
                    break
                
                elif choice.lower() == 'demo':
                    # Hidden demo workflow
                    self.run_demo_workflow()
                    input("\nPress Enter to continue...")
                
                else:
                    print("❌ Invalid choice. Please try again.")
                    time.sleep(1)
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main menu: {e}")
                print(f"❌ Unexpected error: {e}")
                time.sleep(2)


def main():
    """Main application entry point"""
    print("🚀 Starting WorkMate Mobile App Demo")
    print("=" * 60)
    
    # Initialize app
    field_officer_id = input("Enter Field Officer ID (or press Enter for FO001): ").strip() or "FO001"
    app = WorkMateApp(field_officer_id)
    
    print(f"\n📱 Welcome {field_officer_id}!")
    print("💡 Tip: Type 'demo' in the main menu to run automated demo workflow")
    
    # Start main menu
    try:
        app.main_menu()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"💥 Fatal error: {e}")
    
    print("\n🎉 WorkMate App Demo Complete!")

if __name__ == "__main__":
    main() 