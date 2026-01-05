"""UMLS Context Extractor for semantic verification."""

import re
import time
from typing import Any, Dict, List, Optional

import requests

from ..utils.logger import get_logger

logger = get_logger(__name__)


class UMLSContextExtractor:
    """
    Extract semantic context from UMLS Metathesaurus for symptom verification.

    This class queries the UMLS REST API to extract:
    - Synonyms: Lexical variants of the symptom term
    - Variations: Related concepts with qualifiers
    - Associations: Co-occurring symptoms from SNOMED CT
    - Locations: Anatomically related body regions
    - Modifiers: Temporal and exacerbating factors
    """

    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    AUTH_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"

    def __init__(
            self,
            api_key: str,
            version: str = "current",
            synonyms_limit: int = 50,
            variations_limit: int = 25,
            relations_limit: int = 100,
    ):
        """
        Initialize UMLS extractor.

        Args:
            api_key: UMLS API key from NLM
            version: UMLS version (default: "current")
            synonyms_limit: Maximum synonyms to extract
            variations_limit: Maximum variations to extract
            relations_limit: Maximum SNOMED relations to extract
        """
        self.api_key = api_key
        self.version = version
        self.synonyms_limit = synonyms_limit
        self.variations_limit = variations_limit
        self.relations_limit = relations_limit

        self._tgt: Optional[str] = None
        self._tgt_timestamp: float = 0
        self._tgt_lifetime: float = 28800  # 8 hours

    def _get_tgt(self) -> str:
        """Get Ticket Granting Ticket from UMLS."""
        current_time = time.time()

        # Check if TGT is still valid
        if self._tgt and (current_time - self._tgt_timestamp) < self._tgt_lifetime:
            return self._tgt

        logger.debug("Obtaining new TGT from UMLS...")

        response = requests.post(
            self.AUTH_URL,
            data={"apikey": self.api_key}
        )
        response.raise_for_status()

        # Extract TGT from response
        match = re.search(r'api-key/(TGT-[^"]+)', response.text)
        if not match:
            raise ValueError("Failed to extract TGT from UMLS response")

        self._tgt = f"{self.AUTH_URL}/{match.group(1)}"
        self._tgt_timestamp = current_time

        return self._tgt

    def _get_service_ticket(self) -> str:
        """Get a single-use Service Ticket."""
        tgt = self._get_tgt()

        response = requests.post(
            tgt,
            data={"service": "http://umlsks.nlm.nih.gov"}
        )
        response.raise_for_status()

        return response.text

    def search_concept(self, term: str) -> Optional[Dict[str, str]]:
        """
        Search for a UMLS concept by term.

        Args:
            term: Symptom term to search (e.g., "chest pain")

        Returns:
            Dict with 'cui', 'name', and 'source', or None if not found
        """
        ticket = self._get_service_ticket()

        url = f"{self.BASE_URL}/search/{self.version}"
        params = {
            "string": term,
            "ticket": ticket,
            "searchType": "exact",
            "returnIdType": "concept",
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        results = data.get("result", {}).get("results", [])
        if not results:
            # Try word-based search
            params["searchType"] = "words"
            ticket = self._get_service_ticket()
            params["ticket"] = ticket

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("result", {}).get("results", [])

        if results:
            result = results[0]
            return {
                "cui": result["ui"],
                "name": result["name"],
                "source": result.get("rootSource", ""),
            }

        return None

    def get_synonyms(self, cui: str) -> List[str]:
        """
        Get all English synonyms for a CUI.

        Args:
            cui: Concept Unique Identifier

        Returns:
            List of synonym strings
        """
        ticket = self._get_service_ticket()

        url = f"{self.BASE_URL}/content/{self.version}/CUI/{cui}/atoms"
        params = {
            "ticket": ticket,
            "language": "ENG",
            "ttys": "PT,SY,SYN,ET",
            "pageSize": self.synonyms_limit,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        synonyms = []
        for atom in data.get("result", []):
            name = atom.get("name", "")
            if name and name not in synonyms:
                synonyms.append(name)

        return synonyms[:self.synonyms_limit]

    def get_variations(self, term: str) -> List[str]:
        """
        Find related concepts with different qualifiers.

        Args:
            term: Base symptom term

        Returns:
            List of variation strings
        """
        ticket = self._get_service_ticket()

        url = f"{self.BASE_URL}/search/{self.version}"
        params = {
            "string": term,
            "ticket": ticket,
            "searchType": "words",
            "sabs": "SNOMEDCT_US",
            "pageSize": self.variations_limit,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        variations = []
        for result in data.get("result", {}).get("results", []):
            name = result.get("name", "")
            # Only include if it contains the original term
            if name and term.lower() in name.lower() and name not in variations:
                variations.append(name)

        return variations[:self.variations_limit]

    def get_snomed_id(self, cui: str) -> Optional[str]:
        """
        Get SNOMED CT ID from CUI.

        Args:
            cui: Concept Unique Identifier

        Returns:
            SNOMED CT identifier or None
        """
        ticket = self._get_service_ticket()

        url = f"{self.BASE_URL}/content/{self.version}/CUI/{cui}/atoms"
        params = {
            "ticket": ticket,
            "sabs": "SNOMEDCT_US",
            "pageSize": 10,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        for atom in data.get("result", []):
            source_ui = atom.get("sourceUi", "")
            if source_ui:
                return source_ui

        return None

    def get_snomed_relations(self, snomed_id: str) -> Dict[str, List[str]]:
        """
        Get SNOMED CT relations for a concept.

        Args:
            snomed_id: SNOMED CT identifier

        Returns:
            Dict with 'associations', 'locations', and 'modifiers'
        """
        ticket = self._get_service_ticket()

        url = f"{self.BASE_URL}/content/{self.version}/source/SNOMEDCT_US/{snomed_id}/relations"
        params = {
            "ticket": ticket,
            "pageSize": self.relations_limit,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        relations = {
            "associations": [],
            "locations": [],
            "modifiers": [],
        }

        # Keywords for categorization
        association_keywords = ["accompanied", "associated", "with"]
        location_keywords = ["radiating", "left", "right", "sided", "between", "below", "above"]
        modifier_keywords = ["worse", "better", "relieved", "exacerbated", "lasting", "causing"]

        for rel in data.get("result", []):
            name = rel.get("relatedIdName", "")
            if not name:
                continue

            name_lower = name.lower()

            # Categorize based on keywords
            if any(kw in name_lower for kw in modifier_keywords):
                if name not in relations["modifiers"]:
                    relations["modifiers"].append(name)
            elif any(kw in name_lower for kw in location_keywords):
                if name not in relations["locations"]:
                    relations["locations"].append(name)
            elif any(kw in name_lower for kw in association_keywords):
                if name not in relations["associations"]:
                    relations["associations"].append(name)
            else:
                # Default to associations
                if name not in relations["associations"]:
                    relations["associations"].append(name)

        return relations

    def extract_full_context(self, symptom_term: str) -> Optional[Dict[str, Any]]:
        """
        Complete extraction pipeline for a single symptom.

        Args:
            symptom_term: Symptom to extract context for

        Returns:
            Structured semantic context or None if not found
        """
        logger.info(f"Extracting context for: {symptom_term}")

        # Step 1: Find CUI
        concept = self.search_concept(symptom_term)
        if not concept:
            logger.warning(f"No concept found for '{symptom_term}'")
            return None

        cui = concept["cui"]
        logger.debug(f"Found CUI: {cui}")

        # Step 2: Get synonyms
        synonyms = self.get_synonyms(cui)
        logger.debug(f"Found {len(synonyms)} synonyms")

        # Step 3: Get variations
        variations = self.get_variations(symptom_term)
        logger.debug(f"Found {len(variations)} variations")

        # Step 4: Get SNOMED ID and relations
        snomed_id = self.get_snomed_id(cui)
        relations = {"associations": [], "locations": [], "modifiers": []}

        if snomed_id:
            logger.debug(f"Found SNOMED ID: {snomed_id}")
            relations = self.get_snomed_relations(snomed_id)
            logger.debug(
                f"Found {len(relations['associations'])} associations, "
                f"{len(relations['locations'])} locations, "
                f"{len(relations['modifiers'])} modifiers"
            )

        # Assemble final context
        context = {
            "term": symptom_term,
            "cui": cui,
            "snomed_id": snomed_id,
            "synonyms": synonyms,
            "variations": variations,
            "associations": relations["associations"],
            "locations": relations["locations"],
            "modifiers": relations["modifiers"],
        }

        return context


def extract_context_for_patient(
        symptoms: List[str],
        api_key: str,
        **kwargs
) -> Dict[str, Any]:
    """
    Extract semantic context for all symptoms in a patient case.

    Args:
        symptoms: List of symptom terms
        api_key: UMLS API key
        **kwargs: Additional arguments for UMLSContextExtractor

    Returns:
        Dictionary mapping symptom keys to their semantic contexts
    """
    extractor = UMLSContextExtractor(api_key, **kwargs)

    patient_context = {}
    for symptom in symptoms:
        context = extractor.extract_full_context(symptom)
        if context:
            # Use normalized key
            key = symptom.lower().replace(" ", "_").replace("-", "_")
            patient_context[key] = context

    return patient_context
