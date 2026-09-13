import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  IconButton,
  Switch,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  FormControlLabel,
  Alert,
  CircularProgress,
  Tooltip,
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import apiService from '../api/apiService';

const SOURCE_TYPES = [
  { value: 'youtube', label: 'YouTube' },
  { value: 'telegram', label: 'Telegram' },
  { value: 'reddit', label: 'Reddit' },
  { value: 'twitter', label: 'Twitter' },
  { value: 'gnews', label: 'Google News' },
  { value: 'pytrends', label: 'Google Trends' },
];

const PURPOSE_OPTIONS = [
  { value: 'general', label: 'General' },
  { value: 'finance', label: 'Finance' },
];

const SOURCE_COLORS = {
  youtube: 'hsl(0, 85%, 55%)',
  telegram: 'hsl(200, 85%, 55%)',
  reddit: 'hsl(16, 85%, 55%)',
  twitter: 'hsl(200, 95%, 55%)',
  gnews: 'hsl(32, 95%, 55%)',
  pytrends: 'hsl(235, 85%, 65%)',
};

const EMPTY_FORM = {
  name: '',
  source_type: 'youtube',
  credential_ref: '',
  purpose: 'general',
  enabled: true,
  channels: '',
  max_videos_per_channel: 5,
  days_back: 7,
  max_messages: 50,
};

function collectSpecFromForm(form) {
  const spec = {};
  if (form.source_type === 'youtube') {
    if (form.channels) spec.channels = form.channels.split(',').map(c => c.trim()).filter(Boolean);
    spec.max_videos_per_channel = Number(form.max_videos_per_channel) || 5;
    spec.days_back = Number(form.days_back) || 7;
  } else if (form.source_type === 'telegram') {
    if (form.channels) spec.channels = form.channels.split(',').map(c => c.trim()).filter(Boolean);
    spec.max_messages = Number(form.max_messages) || 50;
  } else {
    if (form.channels) spec.channels = form.channels.split(',').map(c => c.trim()).filter(Boolean);
  }
  return spec;
}

function channelsFromConfig(cfg) {
  const channels = cfg.collect_spec?.channels;
  if (!channels) return '';
  return Array.isArray(channels) ? channels.join(', ') : String(channels);
}

function formFromConfig(cfg) {
  return {
    name: cfg.name || '',
    source_type: cfg.source_type || 'youtube',
    credential_ref: cfg.credential_ref || '',
    purpose: cfg.purpose || 'general',
    enabled: cfg.enabled ?? true,
    channels: channelsFromConfig(cfg),
    max_videos_per_channel: cfg.collect_spec?.max_videos_per_channel ?? 5,
    days_back: cfg.collect_spec?.days_back ?? 7,
    max_messages: cfg.collect_spec?.max_messages ?? 50,
  };
}

function ConfigDialog({ open, config, onClose, onSaved }) {
  const isEdit = Boolean(config);
  const [form, setForm] = useState(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (open) {
      setForm(config ? formFromConfig(config) : EMPTY_FORM);
      setError(null);
    }
  }, [open, config]);

  const set = (field, value) => setForm(f => ({ ...f, [field]: value }));

  const handleSave = async () => {
    if (!form.name.trim()) { setError('Name is required'); return; }
    if (!form.credential_ref.trim()) { setError('Credential ref is required'); return; }

    setSaving(true);
    setError(null);
    try {
      const payload = {
        name: form.name.trim(),
        source_type: form.source_type,
        credential_ref: form.credential_ref.trim(),
        purpose: form.purpose,
        enabled: form.enabled,
        collect_spec: collectSpecFromForm(form),
      };
      if (isEdit) {
        await apiService.updateSourceConfig(config.id, payload);
      } else {
        await apiService.createSourceConfig(payload);
      }
      onSaved();
      onClose();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const hasChannels = ['youtube', 'telegram', 'reddit'].includes(form.source_type);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth
      PaperProps={{ sx: { bgcolor: 'hsl(220, 18%, 10%)', border: '1px solid hsl(220, 15%, 18%)' } }}>
      <DialogTitle sx={{ color: 'white', borderBottom: '1px solid hsl(220, 15%, 18%)' }}>
        {isEdit ? 'Edit Source' : 'Add Source'}
      </DialogTitle>
      <DialogContent sx={{ pt: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}

        <TextField
          label="Name" value={form.name} onChange={e => set('name', e.target.value)}
          fullWidth required size="small" placeholder="e.g. Finance YT"
        />

        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl fullWidth size="small">
            <InputLabel>Source Type</InputLabel>
            <Select value={form.source_type} onChange={e => set('source_type', e.target.value)} label="Source Type">
              {SOURCE_TYPES.map(t => <MenuItem key={t.value} value={t.value}>{t.label}</MenuItem>)}
            </Select>
          </FormControl>
          <FormControl fullWidth size="small">
            <InputLabel>Purpose</InputLabel>
            <Select value={form.purpose} onChange={e => set('purpose', e.target.value)} label="Purpose">
              {PURPOSE_OPTIONS.map(p => <MenuItem key={p.value} value={p.value}>{p.label}</MenuItem>)}
            </Select>
          </FormControl>
        </Box>

        <TextField
          label="Credential Ref" value={form.credential_ref} onChange={e => set('credential_ref', e.target.value)}
          fullWidth required size="small" placeholder="e.g. YOUTUBE_API_KEY"
          helperText="Name of the env var holding credentials"
        />

        {hasChannels && (
          <TextField
            label="Channels (comma-separated)" value={form.channels}
            onChange={e => set('channels', e.target.value)}
            fullWidth size="small" multiline minRows={2}
            placeholder="@handle1, @handle2"
          />
        )}

        {form.source_type === 'youtube' && (
          <Box sx={{ display: 'flex', gap: 2 }}>
            <TextField
              label="Max videos / channel" type="number" size="small" fullWidth
              value={form.max_videos_per_channel}
              onChange={e => set('max_videos_per_channel', e.target.value)}
            />
            <TextField
              label="Days back" type="number" size="small" fullWidth
              value={form.days_back}
              onChange={e => set('days_back', e.target.value)}
            />
          </Box>
        )}

        {form.source_type === 'telegram' && (
          <TextField
            label="Max messages" type="number" size="small" fullWidth
            value={form.max_messages}
            onChange={e => set('max_messages', e.target.value)}
          />
        )}

        <FormControlLabel
          control={<Switch checked={form.enabled} onChange={e => set('enabled', e.target.checked)} color="primary" />}
          label="Enabled"
        />
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2, borderTop: '1px solid hsl(220, 15%, 18%)' }}>
        <Button onClick={onClose} disabled={saving}>Cancel</Button>
        <Button variant="contained" onClick={handleSave} disabled={saving}>
          {saving ? <CircularProgress size={18} /> : isEdit ? 'Save' : 'Add'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

function ConfigRow({ config, onEdit, onDeleted, onToggle }) {
  const channels = channelsFromConfig(config);
  const color = SOURCE_COLORS[config.source_type] || 'hsl(220, 15%, 55%)';

  return (
    <Paper sx={{
      p: 2, display: 'flex', alignItems: 'center', gap: 2,
      border: '1px solid hsl(220, 15%, 18%)', borderRadius: 2,
      opacity: config.enabled ? 1 : 0.5,
    }}>
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 0.5 }}>
          <Typography variant="body1" sx={{ fontWeight: 600, color: 'white' }}>
            {config.name}
          </Typography>
          <Chip label={config.source_type} size="small"
            sx={{ bgcolor: `${color}22`, color, border: `1px solid ${color}44`, fontWeight: 600, fontSize: '0.7rem' }} />
          {config.purpose === 'finance' && (
            <Chip label="finance" size="small"
              sx={{ bgcolor: 'hsla(175,85%,50%,0.1)', color: 'hsl(175,85%,50%)', border: '1px solid hsla(175,85%,50%,0.3)', fontSize: '0.7rem' }} />
          )}
        </Box>
        {channels && (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {channels}
          </Typography>
        )}
        <Typography variant="caption" sx={{ color: 'hsl(220,15%,40%)' }}>
          ref: {config.credential_ref}
        </Typography>
      </Box>

      <Tooltip title={config.enabled ? 'Disable' : 'Enable'}>
        <Switch checked={config.enabled} onChange={() => onToggle(config)} size="small" color="primary" />
      </Tooltip>
      <Tooltip title="Edit">
        <IconButton size="small" onClick={() => onEdit(config)} sx={{ color: 'text.secondary', '&:hover': { color: 'primary.main' } }}>
          <EditIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      <Tooltip title="Delete">
        <IconButton size="small" onClick={() => onDeleted(config)} sx={{ color: 'text.secondary', '&:hover': { color: 'error.main' } }}>
          <DeleteIcon fontSize="small" />
        </IconButton>
      </Tooltip>
    </Paper>
  );
}

export default function ChannelConfig() {
  const [configs, setConfigs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState(null);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.listSourceConfigs();
      setConfigs(Array.isArray(data) ? data : []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleEdit = (cfg) => { setEditingConfig(cfg); setDialogOpen(true); };
  const handleAdd = () => { setEditingConfig(null); setDialogOpen(true); };
  const handleDialogClose = () => { setDialogOpen(false); setEditingConfig(null); };

  const handleToggle = async (cfg) => {
    try {
      await apiService.updateSourceConfig(cfg.id, { enabled: !cfg.enabled });
      setConfigs(cs => cs.map(c => c.id === cfg.id ? { ...c, enabled: !c.enabled } : c));
    } catch (e) {
      setError(e.message);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await apiService.deleteSourceConfig(deleteTarget.id);
      setConfigs(cs => cs.filter(c => c.id !== deleteTarget.id));
      setDeleteTarget(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Box>
      <Box sx={{ mb: 4, display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, color: 'white', mb: 0.5 }}>
            Sources
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage channels and data sources
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <IconButton onClick={load} sx={{ color: 'text.secondary' }}>
            <RefreshIcon />
          </IconButton>
          <Button variant="contained" startIcon={<AddIcon />} onClick={handleAdd}>
            Add Source
          </Button>
        </Box>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>{error}</Alert>}

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress sx={{ color: 'primary.main' }} />
        </Box>
      ) : configs.length === 0 ? (
        <Box sx={{
          textAlign: 'center', py: 8, bgcolor: '#1A1A1A', borderRadius: 3,
          border: '1px dashed rgba(255,255,255,0.1)',
        }}>
          <Typography variant="h6" color="text.secondary" gutterBottom>No sources yet</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Add a YouTube channel, Telegram group, or other source
          </Typography>
          <Button variant="contained" startIcon={<AddIcon />} onClick={handleAdd}>
            Add Source
          </Button>
        </Box>
      ) : (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          {configs.map(cfg => (
            <ConfigRow
              key={cfg.id}
              config={cfg}
              onEdit={handleEdit}
              onDeleted={setDeleteTarget}
              onToggle={handleToggle}
            />
          ))}
        </Box>
      )}

      <ConfigDialog
        open={dialogOpen}
        config={editingConfig}
        onClose={handleDialogClose}
        onSaved={load}
      />

      {/* Delete confirm */}
      <Dialog open={Boolean(deleteTarget)} onClose={() => setDeleteTarget(null)}
        PaperProps={{ sx: { bgcolor: 'hsl(220, 18%, 10%)', border: '1px solid hsl(220, 15%, 18%)' } }}>
        <DialogTitle sx={{ color: 'white' }}>Delete source?</DialogTitle>
        <DialogContent>
          <Typography color="text.secondary">
            Remove <strong style={{ color: 'white' }}>{deleteTarget?.name}</strong>? This can't be undone.
          </Typography>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setDeleteTarget(null)} disabled={deleting}>Cancel</Button>
          <Button variant="contained" color="error" onClick={handleDeleteConfirm} disabled={deleting}>
            {deleting ? <CircularProgress size={18} /> : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
