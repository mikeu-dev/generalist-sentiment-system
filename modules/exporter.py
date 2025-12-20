import pandas as pd
import io
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

class Exporter:
    def export_to_excel(self, data, summary_stats):
        """
        Exports data to an Excel file with multiple sheets.
        
        Args:
            data (list): List of dictionaries containing the analysis logs.
            summary_stats (dict): Dictionary containing summary statistics.
            
        Returns:
            BytesIO: The Excel file in memory.
        """
        output = io.BytesIO()
        
        # Create DataFrames
        df_details = pd.DataFrame(data)
        
        # flattening metadata if needed or just use as string
        if 'metadata_json' in df_details.columns:
            df_details['metadata_json'] = df_details['metadata_json'].apply(lambda x: str(x))

        df_stats = pd.DataFrame([summary_stats])

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_stats.to_excel(writer, sheet_name='Summary', index=False)
            df_details.to_excel(writer, sheet_name='Detailed Data', index=False)
            
        output.seek(0)
        return output

    def export_to_pdf(self, data, summary_stats, batch_id):
        """
        Exports data to a PDF report.
        
        Args:
            data (list): List of dictionaries containing the analysis logs.
            summary_stats (dict): Dictionary containing summary statistics.
            batch_id (str): The ID of the analysis batch.
            
        Returns:
            BytesIO: The PDF file in memory.
        """
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = styles['Title']
        story.append(Paragraph(f"Laporan Analisis Sentimen", title_style))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph(f"Batch ID: {batch_id}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Summary Section
        story.append(Paragraph("Ringkasan Statistik", styles['Heading2']))
        
        # Prepare Pie Chart for Distribution
        if 'distribution' in summary_stats:
            dist = summary_stats['distribution']
            # Create a simple figure for the chart
            plt.figure(figsize=(4, 3))
            plt.pie(dist.values(), labels=dist.keys(), autopct='%1.1f%%', startangle=140)
            plt.title('Distribusi Sentimen')
            
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            plt.close()
            
            img = ReportLabImage(img_buf, width=200, height=150)
            story.append(img)
        
        # Stats Table
        stat_data = [['Metric', 'Value']]
        for k, v in summary_stats.items():
            if isinstance(v, dict):
                continue # Skip nested dicts like distribution for the simple table
            stat_data.append([k, str(v)])
            
        t = Table(stat_data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(t)
        story.append(Spacer(1, 24))

        # Detailed Data Sample
        story.append(Paragraph("Sampel Data (Top 50)", styles['Heading2']))
        
        # Table Header
        data_table = [['Teks', 'Sentimen', 'Score']]
        
        # Table Body (Limit to top 50 to avoid PDF explosion)
        for item in data[:50]:
            # Truncate text if too long
            text_preview = (item.get('text', '')[:75] + '..') if len(item.get('text', '')) > 75 else item.get('text', '')
            data_table.append([
                text_preview,
                item.get('label', ''),
                f"{item.get('sentiment_score', 0):.2f}"
            ])
            
        t2 = Table(data_table, colWidths=[300, 100, 80])
        t2.setStyle(TableStyle([
             ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(t2)

        doc.build(story)
        output.seek(0)
        return output
