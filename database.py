"""
Database Module for Counterfeit Drug Detection System

This module provides functions for storing and retrieving authentic medicine information
from a database for verification purposes.
"""

import sqlite3
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import hashlib


class MedicineDatabase:
    """
    Class for managing the authentic medicine database.
    """
    
    def __init__(self, db_path: str = 'medicine_database.db'):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # Create database if it doesn't exist
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """
        Connect to the database.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.cursor = self.conn.cursor()
    
    def _disconnect(self):
        """
        Disconnect from the database.
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def _create_tables(self):
        """
        Create the necessary tables if they don't exist.
        """
        # Create medicines table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS medicines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            manufacturer TEXT NOT NULL,
            ndc TEXT,
            gtin TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create barcodes table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS barcodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_id INTEGER NOT NULL,
            barcode_type TEXT NOT NULL,
            barcode_data TEXT NOT NULL,
            FOREIGN KEY (medicine_id) REFERENCES medicines (id),
            UNIQUE (barcode_data)
        )
        ''')
        
        # Create serial_numbers table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS serial_numbers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_id INTEGER NOT NULL,
            serial_number TEXT NOT NULL,
            batch_number TEXT,
            expiry_date TEXT,
            FOREIGN KEY (medicine_id) REFERENCES medicines (id),
            UNIQUE (serial_number, batch_number)
        )
        ''')
        
        # Create features table for storing image features
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_id INTEGER NOT NULL,
            feature_type TEXT NOT NULL,
            feature_data BLOB NOT NULL,
            FOREIGN KEY (medicine_id) REFERENCES medicines (id)
        )
        ''')
        
        # Create verification_logs table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_id INTEGER,
            barcode_data TEXT,
            serial_number TEXT,
            is_authentic BOOLEAN,
            confidence REAL,
            verification_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            details TEXT
        )
        ''')
        
        self.conn.commit()
    
    def add_medicine(self, medicine_data: Dict[str, Any]) -> int:
        """
        Add a new medicine to the database.
        
        Args:
            medicine_data: Dictionary containing medicine information
            
        Returns:
            ID of the newly added medicine
        """
        # Extract medicine fields
        name = medicine_data.get('name', '')
        manufacturer = medicine_data.get('manufacturer', '')
        ndc = medicine_data.get('ndc', '')
        gtin = medicine_data.get('gtin', '')
        description = medicine_data.get('description', '')
        
        # Insert medicine record
        self.cursor.execute('''
        INSERT INTO medicines (name, manufacturer, ndc, gtin, description)
        VALUES (?, ?, ?, ?, ?)
        ''', (name, manufacturer, ndc, gtin, description))
        
        medicine_id = self.cursor.lastrowid
        
        # Add barcodes if provided
        if 'barcodes' in medicine_data and medicine_data['barcodes']:
            for barcode in medicine_data['barcodes']:
                self.add_barcode(medicine_id, barcode)
        
        # Add serial numbers if provided
        if 'serial_numbers' in medicine_data and medicine_data['serial_numbers']:
            for serial in medicine_data['serial_numbers']:
                self.add_serial_number(medicine_id, serial)
        
        # Add features if provided
        if 'features' in medicine_data and medicine_data['features']:
            for feature_type, feature_data in medicine_data['features'].items():
                self.add_feature(medicine_id, feature_type, feature_data)
        
        self.conn.commit()
        return medicine_id
    
    def add_barcode(self, medicine_id: int, barcode_data: Dict[str, str]) -> int:
        """
        Add a barcode for a medicine.
        
        Args:
            medicine_id: ID of the medicine
            barcode_data: Dictionary containing barcode information
            
        Returns:
            ID of the newly added barcode
        """
        barcode_type = barcode_data.get('type', 'unknown')
        barcode_value = barcode_data.get('data', '')
        
        try:
            self.cursor.execute('''
            INSERT INTO barcodes (medicine_id, barcode_type, barcode_data)
            VALUES (?, ?, ?)
            ''', (medicine_id, barcode_type, barcode_value))
            
            barcode_id = self.cursor.lastrowid
            self.conn.commit()
            return barcode_id
        except sqlite3.IntegrityError:
            # Barcode already exists
            return -1
    
    def add_serial_number(self, medicine_id: int, serial_data: Dict[str, str]) -> int:
        """
        Add a serial number for a medicine.
        
        Args:
            medicine_id: ID of the medicine
            serial_data: Dictionary containing serial number information
            
        Returns:
            ID of the newly added serial number
        """
        serial_number = serial_data.get('serial_number', '')
        batch_number = serial_data.get('batch_number', '')
        expiry_date = serial_data.get('expiry_date', '')
        
        try:
            self.cursor.execute('''
            INSERT INTO serial_numbers (medicine_id, serial_number, batch_number, expiry_date)
            VALUES (?, ?, ?, ?)
            ''', (medicine_id, serial_number, batch_number, expiry_date))
            
            serial_id = self.cursor.lastrowid
            self.conn.commit()
            return serial_id
        except sqlite3.IntegrityError:
            # Serial number already exists
            return -1
    
    def add_feature(self, medicine_id: int, feature_type: str, feature_data: Any) -> int:
        """
        Add feature data for a medicine.
        
        Args:
            medicine_id: ID of the medicine
            feature_type: Type of feature (e.g., 'color_hist', 'deep', 'hog')
            feature_data: Feature data (will be serialized to JSON)
            
        Returns:
            ID of the newly added feature
        """
        # Serialize feature data to JSON
        serialized_data = json.dumps(feature_data)
        
        self.cursor.execute('''
        INSERT INTO features (medicine_id, feature_type, feature_data)
        VALUES (?, ?, ?)
        ''', (medicine_id, feature_type, serialized_data))
        
        feature_id = self.cursor.lastrowid
        self.conn.commit()
        return feature_id
    
    def verify_barcode(self, barcode_data: str) -> Dict[str, Any]:
        """
        Verify if a barcode exists in the database.
        
        Args:
            barcode_data: Barcode data to verify
            
        Returns:
            Dictionary with verification results
        """
        self.cursor.execute('''
        SELECT m.*, b.barcode_type
        FROM medicines m
        JOIN barcodes b ON m.id = b.medicine_id
        WHERE b.barcode_data = ?
        ''', (barcode_data,))
        
        row = self.cursor.fetchone()
        
        if row:
            # Convert row to dictionary
            medicine = dict(row)
            
            # Get all barcodes for this medicine
            self.cursor.execute('''
            SELECT * FROM barcodes WHERE medicine_id = ?
            ''', (medicine['id'],))
            
            barcodes = [dict(b) for b in self.cursor.fetchall()]
            medicine['barcodes'] = barcodes
            
            # Get all serial numbers for this medicine
            self.cursor.execute('''
            SELECT * FROM serial_numbers WHERE medicine_id = ?
            ''', (medicine['id'],))
            
            serials = [dict(s) for s in self.cursor.fetchall()]
            medicine['serial_numbers'] = serials
            
            return {
                'verified': True,
                'medicine': medicine,
                'confidence': 1.0
            }
        else:
            # Check for similar barcodes (fuzzy matching)
            self.cursor.execute('''
            SELECT barcode_data FROM barcodes
            ''')
            
            all_barcodes = [row[0] for row in self.cursor.fetchall()]
            
            # Find most similar barcode
            most_similar = None
            highest_similarity = 0
            
            for db_barcode in all_barcodes:
                similarity = self._calculate_string_similarity(barcode_data, db_barcode)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar = db_barcode
            
            if highest_similarity > 0.8:
                # If similarity is high, get the medicine for this barcode
                self.cursor.execute('''
                SELECT m.*, b.barcode_type
                FROM medicines m
                JOIN barcodes b ON m.id = b.medicine_id
                WHERE b.barcode_data = ?
                ''', (most_similar,))
                
                row = self.cursor.fetchone()
                medicine = dict(row) if row else None
                
                return {
                    'verified': False,
                    'similar_barcode': most_similar,
                    'similarity': highest_similarity,
                    'medicine': medicine,
                    'confidence': highest_similarity
                }
            
            return {
                'verified': False,
                'confidence': 0.0
            }
    
    def verify_serial_number(self, serial_number: str, batch_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify if a serial number exists in the database.
        
        Args:
            serial_number: Serial number to verify
            batch_number: Batch number (optional)
            
        Returns:
            Dictionary with verification results
        """
        if batch_number:
            # If batch number is provided, check both
            self.cursor.execute('''
            SELECT m.*, s.batch_number, s.expiry_date
            FROM medicines m
            JOIN serial_numbers s ON m.id = s.medicine_id
            WHERE s.serial_number = ? AND s.batch_number = ?
            ''', (serial_number, batch_number))
        else:
            # Otherwise, check just the serial number
            self.cursor.execute('''
            SELECT m.*, s.batch_number, s.expiry_date
            FROM medicines m
            JOIN serial_numbers s ON m.id = s.medicine_id
            WHERE s.serial_number = ?
            ''', (serial_number,))
        
        row = self.cursor.fetchone()
        
        if row:
            # Convert row to dictionary
            medicine = dict(row)
            
            # Get all barcodes for this medicine
            self.cursor.execute('''
            SELECT * FROM barcodes WHERE medicine_id = ?
            ''', (medicine['id'],))
            
            barcodes = [dict(b) for b in self.cursor.fetchall()]
            medicine['barcodes'] = barcodes
            
            # Get all serial numbers for this medicine
            self.cursor.execute('''
            SELECT * FROM serial_numbers WHERE medicine_id = ?
            ''', (medicine['id'],))
            
            serials = [dict(s) for s in self.cursor.fetchall()]
            medicine['serial_numbers'] = serials
            
            return {
                'verified': True,
                'medicine': medicine,
                'confidence': 1.0
            }
        else:
            # Check for similar serial numbers (fuzzy matching)
            self.cursor.execute('''
            SELECT serial_number FROM serial_numbers
            ''')
            
            all_serials = [row[0] for row in self.cursor.fetchall()]
            
            # Find most similar serial number
            most_similar = None
            highest_similarity = 0
            
            for db_serial in all_serials:
                similarity = self._calculate_string_similarity(serial_number, db_serial)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar = db_serial
            
            if highest_similarity > 0.8:
                # If similarity is high, get the medicine for this serial number
                self.cursor.execute('''
                SELECT m.*, s.batch_number, s.expiry_date
                FROM medicines m
                JOIN serial_numbers s ON m.id = s.medicine_id
                WHERE s.serial_number = ?
                ''', (most_similar,))
                
                row = self.cursor.fetchone()
                medicine = dict(row) if row else None
                
                return {
                    'verified': False,
                    'similar_serial': most_similar,
                    'similarity': highest_similarity,
                    'medicine': medicine,
                    'confidence': highest_similarity
                }
            
            return {
                'verified': False,
                'confidence': 0.0
            }
    
    def log_verification(self, verification_data: Dict[str, Any]):
        """
        Log a verification attempt.
        
        Args:
            verification_data: Dictionary containing verification information
        """
        medicine_id = verification_data.get('medicine_id')
        barcode_data = verification_data.get('barcode_data', '')
        serial_number = verification_data.get('serial_number', '')
        is_authentic = verification_data.get('is_authentic', False)
        confidence = verification_data.get('confidence', 0.0)
        details = json.dumps(verification_data.get('details', {}))
        
        self.cursor.execute('''
        INSERT INTO verification_logs 
        (medicine_id, barcode_data, serial_number, is_authentic, confidence, details)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (medicine_id, barcode_data, serial_number, is_authentic, confidence, details))
        
        self.conn.commit()
    
    def get_medicine_by_id(self, medicine_id: int) -> Optional[Dict[str, Any]]:
        """
        Get medicine information by ID.
        
        Args:
            medicine_id: ID of the medicine
            
        Returns:
            Dictionary containing medicine information or None if not found
        """
        self.cursor.execute('''
  
(Content truncated due to size limit. Use line ranges to read in chunks)